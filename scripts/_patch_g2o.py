"""Patch g2o HyperGraph destructor to not call clear() (prevents double-free with OptimizableGraph)."""
path = "/root/autodl-tmp/slam_ws/ORB_SLAM2/Thirdparty/g2o/g2o/core/hyper_graph.cpp"
with open(path) as f:
    src = f.read()

old = "  HyperGraph::~HyperGraph()\n  {\n    clear();\n  }"
new = "  HyperGraph::~HyperGraph()\n  {\n    // clear() is called by OptimizableGraph::~OptimizableGraph();\n    // calling it here too causes double-free.\n  }"

if old in src:
    src = src.replace(old, new, 1)
    with open(path, "w") as f:
        f.write(src)
    print("PATCHED OK")
else:
    # Try to find what's actually there
    idx = src.find("HyperGraph::~HyperGraph")
    print("NOT FOUND, context:", repr(src[idx-2:idx+80]) if idx >= 0 else "not found at all")
