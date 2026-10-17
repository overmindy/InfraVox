"""Bound RMSNorm squares for HTP FP16 math while preserving denominator units."""
import copy


def stabilize(model,scale_floor=1.0):
    import numpy as np
    import onnx
    from onnx import helper,numpy_helper
    producers={v:n for n in model.graph.node for v in n.output}
    prefixes=("/norm1/","/norm2/","/norm1_1/","/norm2_1/")
    before,after,records={},{},[]
    opset=next(v.version for v in model.opset_import if v.domain in ("","ai.onnx"))
    for index,prefix in enumerate(prefixes):
        reductions=[n for n in model.graph.node if n.name.startswith(prefix)and n.op_type=="ReduceMean"]
        if len(reductions)!=1:
            raise ValueError("Expected one RMS reduction per prefix")
        reduce=reductions[0];square=producers[reduce.input[0]]
        if square.op_type=="Pow":
            x=square.input[0]
        elif square.op_type=="Mul" and square.input[0]==square.input[1]:
            x=square.input[0]
        else:
            raise ValueError("Expected a square before the RMS reduction")
        additions=[n for n in model.graph.node if n.op_type=="Add"and reduce.output[0] in n.input]
        if len(additions)>1:
            raise ValueError("Ambiguous RMS epsilon addition")
        add=additions[0]if additions else None
        rms_input=add.output[0]if add else reduce.output[0]
        roots=[n for n in model.graph.node if n.op_type=="Sqrt"and rms_input in n.input]
        if len(roots)!=1:
            raise ValueError("Expected one RMS square root")
        root=roots[0]
        tag=f"moshi_stable_rms_{index}"
        if not 0<scale_floor<=1:
            raise ValueError("RMS scale floor must be in (0,1]")
        model.graph.initializer.append(numpy_helper.from_array(np.array(scale_floor,np.float32),tag+"_floor"))
        max_inputs=[tag+"_abs"]
        if opset>=18:
            model.graph.initializer.append(numpy_helper.from_array(np.array([-1],np.int64),tag+"_axes"))
            max_inputs.append(tag+"_axes")
        maximum=helper.make_node("ReduceMax",max_inputs,[tag+"_max"],name=tag+"_max_op",keepdims=1,**({}if opset>=18 else {"axes":[-1]}))
        inserted=[helper.make_node("Abs",[x],[tag+"_abs"],name=tag+"_abs_op"),maximum,
                  helper.make_node("Max",[tag+"_max",tag+"_floor"],[tag+"_scale"],name=tag+"_scale_op"),
                  helper.make_node("Div",[x,tag+"_scale"],[tag+"_bounded"],name=tag+"_bounded_op")]
        square.input[0]=tag+"_bounded"
        if square.op_type=="Mul":square.input[1]=tag+"_bounded"
        before[square.name]=inserted
        if add:
            epsilon=next(v for v in add.input if v!=reduce.output[0])
            before[add.name]=[helper.make_node("Div",[epsilon,tag+"_scale"],[tag+"_eps1"],name=tag+"_eps1_op"),
                              helper.make_node("Div",[tag+"_eps1",tag+"_scale"],[tag+"_eps2"],name=tag+"_eps2_op")]
            for i,name in enumerate(add.input):
                if name==epsilon:add.input[i]=tag+"_eps2"
        output=root.output[0]
        root.output[0]=tag+"_rms_bounded"
        after[root.name]=[helper.make_node("Mul",[tag+"_rms_bounded",tag+"_scale"],[output],name=tag+"_restore_units")]
        records.append({"prefix":prefix,"square":square.name,"input":x,"epsilon_present":add is not None,
                        "scale_floor":scale_floor,"identity":"sqrt(mean((x/s)^2) + epsilon/s/s) * s; s=max(max(abs(x)),floor)"})
    nodes=[]
    for node in model.graph.node:
        nodes.extend(before.get(node.name,[]));nodes.append(copy.deepcopy(node));nodes.extend(after.get(node.name,[]))
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    return records
