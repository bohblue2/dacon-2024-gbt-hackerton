git clone https://github.com/bohblue2/dacon-2024-gbt-hackerton

sudo apt-get install unzip -y
if [ -f /etc/zsh/zshrc ]; then
    echo "ZSH detected"
    conda init zsh && source ~/.zshrc
elif [ -f ~/.bashrc ]; then
    echo "Bash detected"
    conda init bash && source ~/.bashrc
else
    echo "Neither ZSH nor Bash detected, defaulting to Bash"
    conda init bash && source ~/.bashrc
fi

if command -v tmux &> /dev/null; then
    echo "tmux is installed"
else
    echo "tmux is not installed"
fi
conda activate base
conda install -c huggingface transformers huggingface_hub scikit-learn -y
pip install uv

# waandb 
uv pip install wandb
wandb login --cloud ba1de0b3393dcd888d28c0f8fa5deef027df47e6

# Dacon
wget https://bit.ly/3gMPScE -O dacon_submit_api-0.0.4-py3-none-any.zip
unzip dacon_submit_api-0.0.4-py3-none-any.whl 
pip install dacon_submit_api-0.0.4-py3-none-any.whl
rm -rf dacon_submit_api-0.0.4-py3-none-any.whl 