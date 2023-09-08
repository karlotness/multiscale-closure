#!/bin/bash

set -euxo pipefail

module load ffmpeg/4.4.2-nix

for scale in '64' '48'; do
    for size in '100' '50' '25' '10' '2'; do
        pushd .
        cd "sample_size${size}-scale${scale}_jetonly"
        ffmpeg -r 15 -f image2 -i frame%06d.png -vcodec libx264 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -crf 25  -pix_fmt yuv420p "sample_size${size}-scale${scale}_jetonly.mp4"
        popd || exit 1
    done
done
