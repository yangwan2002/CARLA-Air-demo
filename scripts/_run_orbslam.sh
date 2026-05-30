#!/bin/bash
pkill -f rgbd_tum 2>/dev/null
sleep 1
cd /root/autodl-tmp/CARLA-Air-demo
rm -f orbslam2_ugv.log CameraTrajectory.txt KeyFrameTrajectory.txt

EXE=/root/autodl-tmp/slam_ws/ORB_SLAM2/Examples/RGB-D/rgbd_tum
VOC=/root/autodl-tmp/slam_ws/ORB_SLAM2/Vocabulary/ORBvoc.txt
CFG=/root/autodl-tmp/CARLA-Air-demo/CARLA_UGV_orbslam2.yaml
SEQ=/root/autodl-tmp/CARLA-Air-demo/AirGroundRelay-Sim/sequences/paper_eval_l2_slam_rgbd
ASSOC=$SEQ/ugv/association.txt
LOG=/root/autodl-tmp/CARLA-Air-demo/orbslam2_ugv.log

echo "Starting ORB-SLAM2 RGBD (no_viewer, tcmalloc) ..."
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4 \
$EXE $VOC $CFG $SEQ $ASSOC no_viewer 2>&1 | tee $LOG
echo "Exit code: $?"
echo "--- output files ---"
ls -lh /root/autodl-tmp/CARLA-Air-demo/CameraTrajectory.txt \
        /root/autodl-tmp/CARLA-Air-demo/KeyFrameTrajectory.txt 2>/dev/null
wc -l /root/autodl-tmp/CARLA-Air-demo/CameraTrajectory.txt 2>/dev/null
