# This bash imitates "setx" on windows - permanently set environment variable on Ubuntu (18.x.x+)
sudo touch ~/.bashrc
sudo chmod +w ~/.bashrc
sudo sed -i "/^\s*export\s*$1\s*=/d" ~/.bashrc
echo "export $1=\"$2\"" | sudo tee -a ~/.bashrc >/dev/null
export $1=$2
