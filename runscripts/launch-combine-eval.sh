#! /bin/bash
set -euo pipefail

readonly downscale128to96_small='/scratch/kto236/closure/run_outputs/multi-train-cnn-20230202-171821-downscale128to96-small-1-29792466/weights/best_loss.eqx /scratch/kto236/closure/run_outputs/multi-train-cnn-20230202-171821-downscale128to96-small-2-29792539/weights/best_loss.eqx /scratch/kto236/closure/run_outputs/multi-train-cnn-20230202-171821-downscale128to96-small-3-29792572/weights/best_loss.eqx'
readonly downscale128to96_medium='/scratch/kto236/closure/run_outputs/multi-train-cnn-20230202-171821-downscale128to96-medium-1-29792605/weights/best_loss.eqx /scratch/kto236/closure/run_outputs/multi-train-cnn-20230202-171821-downscale128to96-medium-2-29792638/weights/best_loss.eqx /scratch/kto236/closure/run_outputs/multi-train-cnn-20230202-171821-downscale128to96-medium-3-29792671/weights/best_loss.eqx'
readonly downscale128to64_small='/scratch/kto236/closure/run_outputs/multi-train-cnn-20230202-171821-downscale128to64-small-1-29792469/weights/best_loss.eqx /scratch/kto236/closure/run_outputs/multi-train-cnn-20230202-171821-downscale128to64-small-2-29792542/weights/best_loss.eqx /scratch/kto236/closure/run_outputs/multi-train-cnn-20230202-171821-downscale128to64-small-3-29792575/weights/best_loss.eqx'
readonly downscale128to64_medium='/scratch/kto236/closure/run_outputs/multi-train-cnn-20230202-171821-downscale128to64-medium-1-29792608/weights/best_loss.eqx /scratch/kto236/closure/run_outputs/multi-train-cnn-20230202-171821-downscale128to64-medium-2-29792641/weights/best_loss.eqx /scratch/kto236/closure/run_outputs/multi-train-cnn-20230202-171821-downscale128to64-medium-3-29792674/weights/best_loss.eqx'
readonly downscale96to64_small='/scratch/kto236/closure/run_outputs/multi-train-cnn-20230202-171821-downscale96to64-small-1-29792472/weights/best_loss.eqx /scratch/kto236/closure/run_outputs/multi-train-cnn-20230202-171821-downscale96to64-small-2-29792545/weights/best_loss.eqx /scratch/kto236/closure/run_outputs/multi-train-cnn-20230202-171821-downscale96to64-small-3-29792578/weights/best_loss.eqx'
readonly downscale96to64_medium='/scratch/kto236/closure/run_outputs/multi-train-cnn-20230202-171821-downscale96to64-medium-1-29792611/weights/best_loss.eqx /scratch/kto236/closure/run_outputs/multi-train-cnn-20230202-171821-downscale96to64-medium-2-29792644/weights/best_loss.eqx /scratch/kto236/closure/run_outputs/multi-train-cnn-20230202-171821-downscale96to64-medium-3-29792677/weights/best_loss.eqx'
readonly buildup64to128_small='/scratch/kto236/closure/run_outputs/multi-train-cnn-20230202-171821-buildup64to128-small-1-29792487/weights/best_loss.eqx /scratch/kto236/closure/run_outputs/multi-train-cnn-20230202-171821-buildup64to128-small-2-29792560/weights/best_loss.eqx /scratch/kto236/closure/run_outputs/multi-train-cnn-20230202-171821-buildup64to128-small-3-29792593/weights/best_loss.eqx'
readonly buildup64to128_medium='/scratch/kto236/closure/run_outputs/multi-train-cnn-20230202-171821-buildup64to128-medium-1-29792626/weights/best_loss.eqx /scratch/kto236/closure/run_outputs/multi-train-cnn-20230202-171821-buildup64to128-medium-2-29792659/weights/best_loss.eqx /scratch/kto236/closure/run_outputs/multi-train-cnn-20230202-171821-buildup64to128-medium-3-29792692/weights/best_loss.eqx'
readonly buildup96to128_small='/scratch/kto236/closure/run_outputs/multi-train-cnn-20230202-171821-buildup96to128-small-1-29792484/weights/best_loss.eqx /scratch/kto236/closure/run_outputs/multi-train-cnn-20230202-171821-buildup96to128-small-2-29792557/weights/best_loss.eqx /scratch/kto236/closure/run_outputs/multi-train-cnn-20230202-171821-buildup96to128-small-3-29792590/weights/best_loss.eqx'
readonly buildup96to128_medium='/scratch/kto236/closure/run_outputs/multi-train-cnn-20230202-171821-buildup96to128-medium-1-29792623/weights/best_loss.eqx /scratch/kto236/closure/run_outputs/multi-train-cnn-20230202-171821-buildup96to128-medium-2-29792656/weights/best_loss.eqx /scratch/kto236/closure/run_outputs/multi-train-cnn-20230202-171821-buildup96to128-medium-3-29792689/weights/best_loss.eqx'
readonly buildup64to96_small='/scratch/kto236/closure/run_outputs/multi-train-cnn-20230202-171821-buildup64to96-small-1-29792490/weights/best_loss.eqx /scratch/kto236/closure/run_outputs/multi-train-cnn-20230202-171821-buildup64to96-small-2-29792563/weights/best_loss.eqx /scratch/kto236/closure/run_outputs/multi-train-cnn-20230202-171821-buildup64to96-small-3-29792596/weights/best_loss.eqx'
readonly buildup64to96_medium='/scratch/kto236/closure/run_outputs/multi-train-cnn-20230202-171821-buildup64to96-medium-1-29792629/weights/best_loss.eqx /scratch/kto236/closure/run_outputs/multi-train-cnn-20230202-171821-buildup64to96-medium-2-29792662/weights/best_loss.eqx /scratch/kto236/closure/run_outputs/multi-train-cnn-20230202-171821-buildup64to96-medium-3-29792695/weights/best_loss.eqx'

function abbrev_arch() {
    local ARCH_NAME="$1"
    if [[ "$ARCH_NAME" == "small" ]]; then
        echo "sm"
        return 0
    elif [[ "$ARCH_NAME" == "medium" ]]; then
        echo "md"
        return 0
    elif [[ "$ARCH_NAME" == "large" ]]; then
        echo "lg"
        return 0
    fi
}

# 128 -> 96 -> 128
for small_size in 64 96 128; do
    for big_size in 64 96 128; do
        if (( small_size >= big_size )); then
            continue
        fi
        for buildup_arch in small medium; do
            buildup_abbrev=$(abbrev_arch "$buildup_arch")
            buildup_var="buildup${small_size}to${big_size}_${buildup_arch}"
            for downscale_arch in small medium; do
                downscale_abbrev=$(abbrev_arch "$downscale_arch")
                downscale_var="downscale${big_size}to${small_size}_${downscale_arch}"
                sbatch combine-eval-cnn.sh "${small_size}to${big_size}-${downscale_abbrev}${buildup_abbrev}" "${!downscale_var}" "${!buildup_var}"
            done
        done
    done
done
