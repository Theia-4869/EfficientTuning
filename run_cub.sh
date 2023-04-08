#!/bin/bash
#SBATCH --job-name=subset_ablation_grad_square
#SBATCH -o output.out
#SBATCH -e error.out
#SBATCH --partition=q_intel_gpu_nvidia 
#SBATCH --gres=gpu:1

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/online1/public/support/amd/Ananconda3/2022.10/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/online1/public/support/amd/Ananconda3/2022.10/etc/profile.d/conda.sh" ]; then
        . "/online1/public/support/amd/Ananconda3/2022.10/etc/profile.d/conda.sh"
    else
        export PATH="/online1/public/support/amd/Ananconda3/2022.10/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

conda activate prompt
bash scripts/tune_cub.sh