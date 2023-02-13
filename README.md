# droneGazeboGym
## Overview 
(it currently does not work)
## Installation
### Step 1 - Ensure you have Ubuntu 18.04 (e.g. through a virtual machine)
### Step 2 - Upgrade python to python 3.7 (https://cloudbytes.dev/snippets/upgrade-python-to-latest-version-on-ubuntu-linux):
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update -y
#apt list | grep python3.7
sudo apt install python3.7 -y
sudo nano /usr/bin/gnome-terminal
change #!/usr/bin/python3 to #!/usr/bin/python3.7
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 1
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 2
sudo update-alternatives --config python3
(select option 2)
sudo apt remove --purge python3-apt -y
sudo apt autoclean
sudo apt install python3-apt
sudo apt install python3.7-distutils -y
sudo apt install curl
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
sudo python3.7 get-pip.py
### Step 3 - Install ROS Melodic (http://wiki.ros.org/melodic/Installation/Ubuntu):
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
sudo apt install curl
curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
sudo apt-get update -y
sudo apt install ros-melodic-desktop-full -y
echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc
source ~/.bashrc
### Step 4 - Various installs and create a catkin workspace:
#Various installs:
sudo apt install python3-pip python3-all-dev python3-rospkg -y
sudo apt install ros-melodic-desktop-full --fix-missing -y
sudo apt update
sudo apt-get install python3-catkin-tools -y
sudo apt install git
sudo apt upgrade libignition-math2 -y
pip3 install pyulog
pip3 install future
sudo snap install sublime-text --classic
sudo pip3 install opencv-python
pip3 install scipy
pip3 install gym==0.15.7
pip3 install torch
#Create a catkin workspace:
source /opt/ros/melodic/setup.bash
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/
catkin build
source devel/setup.bash
echo $ROS_PACKAGE_PATH
### Step 5 - Get the Gazebo Model for the Uvify IFO-S (https://github.com/decargroup/ifo_gazebo):
cd ~/catkin_ws/src
git clone https://github.com/decarsg/ifo_gazebo.git --recursive
cd ..
catkin config --blacklist px4
catkin build
catkin build
cd ..
bash ~/catkin_ws/src/ifo_gazebo/PX4-Autopilot/Tools/setup/ubuntu.sh
Restart computer

cd ~/catkin_ws/src/ifo_gazebo/PX4-Autopilot
make distclean

cd ~
pip3 install --user empy
pip3 install --user packaging
pip3 install --user toml
pip3 install --user numpy
pip3 install --user jinja2

cd ~/catkin_ws/src/ifo_gazebo/PX4-Autopilot
make px4_sitl gazebo
#cntrl+c
make px4_sitl gazebo
#cntrl+c

cd ~/catkin_ws/src/ifo_gazebo
rm -r real*
git clone https://github.com/pal-robotics/realsense_gazebo_plugin.git
cd ~/catkin_ws
catkin build
catkin build

cd ~
wget https://raw.githubusercontent.com/PX4/Devguide/master/build_scripts/ubuntu_sim_ros_melodic.sh
bash ubuntu_sim_ros_melodic.sh
sudo rosdep init
rosdep update
bash ubuntu_sim_ros_melodic.sh

echo "source ~/catkin_ws/devel/setup.bash" >> ~/.bashrc
echo "source ~/catkin_ws/src/ifo_gazebo/setup_ifo_gazebo.bash suppress" >> ~/.bashrc
cd ~/catkin_ws
source ~/.bashrc
roslaunch ifo_gazebo ifo_empty_world_1.launch
### Step 6 - ROS package that allows a user to communicate with PX4 autopilot using MAVROS (https://github.com/troiwill/mavros-px4-vehicle):
cd ~/catkin_ws/src
git clone https://github.com/troiwill/mavros-px4-vehicle.git
chmod +x mavros-px4-vehicle/scripts/*.py
chmod +x mavros-px4-vehicle/test/*.py
ln -s mavros-px4-vehicle mavros-px4-vehicle
cd ~/catkin_ws
catkin build
source devel/setup.bash