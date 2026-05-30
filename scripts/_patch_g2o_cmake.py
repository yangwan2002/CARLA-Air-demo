"""Add -std=c++14 to g2o CMakeLists.txt so it matches ORB-SLAM2's main code standard."""
path = "/root/autodl-tmp/slam_ws/ORB_SLAM2/Thirdparty/g2o/CMakeLists.txt"
with open(path) as f:
    src = f.read()

old = 'SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${g2o_CXX_FLAGS}")'
new = 'SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${g2o_CXX_FLAGS} -std=c++14")'

if old in src:
    src = src.replace(old, new, 1)
    with open(path, "w") as f:
        f.write(src)
    print("PATCHED OK")
    print("Result:", [l for l in src.splitlines() if "std=c++" in l or "g2o_CXX_FLAGS" in l][:5])
else:
    print("NOT FOUND - looking for alternative...")
    for line in src.splitlines():
        if "CXX_FLAGS" in line:
            print(repr(line))
