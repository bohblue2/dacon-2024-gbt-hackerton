if [ -f /etc/zsh/zshrc ]; then
    echo "ZSH detected"
    conda init zsh && source /root/.zshrc && conda activate base
elif [ -f ~/.bashrc ]; then
    echo "Bash detected"
    conda init bash && source /root/.bashrc && conda activate base
else
    echo "Neither ZSH nor Bash detected, defaulting to Bash"
    conda init bash && source /root/.bashrc  && conda activate base
fi

sudo apt-get install unzip -y

# Python env setup
conda activate base
conda install -c huggingface transformers huggingface_hub scikit-learn -y
pip install uv && uv pip install wandb pgwalker
# NOTE: wandb login with your api key 

# Dacon
wget https://bit.ly/3gMPScE -O dacon_submit_api-0.0.4-py3-none-any.zip
unzip dacon_submit_api-0.0.4-py3-none-any.zip
pip install dacon_submit_api-0.0.4-py3-none-any.whl
rm -rf dacon_submit_api-0.0.4-py3-none-any.whl dacon_submit_api-0.0.4-py3-none-any.zip