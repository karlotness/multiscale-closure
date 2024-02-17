set -euxo pipefail
shopt -s failglob

readonly JOBTMP_DIR="$(mktemp --tmpdir="/host_scratch" -d "apptbuild_XXXXXXXXXX")"
readonly OUTPUT_DIR="$(mktemp --tmpdir="/host_scratch" -d "container_out_XXXXXXXXXX")"

function cleanup_jobtmp_dir() {
    rm -rf "$JOBTMP_DIR"
}
trap cleanup_jobtmp_dir EXIT

# Create filesystem to use for build
FS_FILE="${JOBTMP_DIR}/buildfs.ext4"
truncate -s 30G "$FS_FILE"
mkfs.ext4 "$FS_FILE"
sudo mkdir -p /mnt/buildfs/
sudo mount -o loop "$FS_FILE" /mnt/buildfs
sudo chmod -R 777 /mnt/buildfs

cd /vagrant
APPTAINER_CACHEDIR="${JOBTMP_DIR}/apptainer" APPTAINER_TMPDIR="/mnt/buildfs" apptainer build "${OUTPUT_DIR}/closure.sif" closure.def

sudo umount /mnt/buildfs
sudo rmdir /mnt/buildfs

printf 'Container at %s/closure.sif\n' "$OUTPUT_DIR"
printf 'HASHES\n'
sha1sum "${OUTPUT_DIR}/closure.sif"
