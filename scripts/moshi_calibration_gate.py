"""Evaluate mixed projection weights on retained inputs and shard recurrence."""
import json
from pathlib import Path

from moshi_quantize_temporal_offline import sha256


class CalibrationGate:
    def __init__(self,reference_onnx,probe_onnx,state,limit=0.095):
        import numpy as np
        import onnxruntime as ort
        self.np=np
        self.limit=limit
        self.state=state
        opts=ort.SessionOptions()
        opts.intra_op_num_threads=4
        opts.inter_op_num_threads=1
        reference=ort.InferenceSession(str(reference_onnx),opts,providers=["CPUExecutionProvider"])
        self.probe=ort.InferenceSession(str(probe_onnx),opts,providers=["CPUExecutionProvider"])
        self.names=[v.name for v in reference.get_outputs()]
        self.cache_names=[v.name for v in reference.get_inputs() if v.name.startswith("layer_")]
        self.feeds,self.targets=[],[]
        for sample in state["samples"]:
            path=Path(state["calibration_manifest"]).parent/sample["file"]
            if sha256(path)!=sample["sha256"]:
                raise ValueError("Captured shard input changed")
            with np.load(path) as data:
                feed={v.name:data[v.name] for v in reference.get_inputs()}
            for name in self.cache_names:
                feed[name]=feed[name][:,:,:1000,:].copy()
            self.feeds.append(feed)
            self.targets.append(reference.run(None,feed))
        if len(self.feeds)!=16 or len(self.cache_names)!=4:
            raise ValueError("Require sixteen samples and the four-cache delta ABI")
        self.recurrent_targets={}
        self.sequences=[]
        for source_id in dict.fromkeys(v["source_id"] for v in state["samples"]):
            indices=[next(i for i,v in enumerate(state["samples"]) if v["source_id"]==source_id and v["frame"]==f) for f in range(3)]
            self.sequences.append(indices)
            states={n:self.feeds[indices[0]][n].copy() for n in self.cache_names}
            for frame,index in enumerate(indices):
                actual=reference.run(None,dict(self.feeds[index],**states))
                self.recurrent_targets[index]=actual
                for name in self.cache_names:
                    states[name][:,:,frame:frame+1,:]=actual[self.names.index("output_"+name)]

    def compare(self,actual,expected):
        np=self.np
        errors={n:float(np.linalg.norm(a.astype(np.float64)-b)/max(np.linalg.norm(b.astype(np.float64)),1e-12))
                for n,a,b in zip(self.names,actual,expected)}
        passed=max(errors.values())<=self.limit and all(np.isfinite(a).all() and (np.any(a) or not np.any(b)) for a,b in zip(actual,expected))
        return errors,passed

    def __call__(self,weights):
        records=[]
        for index in [1,9,2,10,15,7,5,13,0,8,3,4,6,11,12,14]:
            actual=self.probe.run(None,dict(self.feeds[index],**weights))
            errors,passed=self.compare(actual,self.targets[index])
            records.append({"mode":"teacher","sample_index":index,"relative_rms":errors,"passed":passed})
            if not passed:
                return {"passed":False,"samples":records}
        for indices in self.sequences:
            states={n:self.feeds[indices[0]][n].copy() for n in self.cache_names}
            for frame,index in enumerate(indices):
                actual=self.probe.run(None,dict(self.feeds[index],**states,**weights))
                errors,passed=self.compare(actual,self.recurrent_targets[index])
                records.append({"mode":"recurrent","sample_index":index,"relative_rms":errors,"passed":passed})
                if not passed:
                    return {"passed":False,"samples":records}
                for name in self.cache_names:
                    states[name][:,:,frame:frame+1,:]=actual[self.names.index("output_"+name)]
        return {"passed":True,"samples":records}
