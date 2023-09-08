import zarr
import pathlib
import argparse

parser = argparse.ArgumentParser(description="Train neural networks for closure")
parser.add_argument("src_dir", type=str, help="Directory containing input zarr files")
parser.add_argument("dest_dir", type=str, help="Directory to store packed zip files")

def main():
    args = parser.parse_args()
    in_dir = pathlib.Path(args.src_dir)
    out_dir = pathlib.Path(args.dest_dir)
    if not in_dir.is_dir():
        raise ValueError(f"Input directory does not exist {in_dir}")
    for zarr_dir_store in in_dir.glob("*/*.zarr"):
        zarr_zip_store = out_dir / (zarr_dir_store.relative_to(in_dir).parent) / (f"{zarr_dir_store.name}.zip")
        print(f"Processing {zarr_dir_store}")
        print(f"    Repacking to {zarr_zip_store}")
        zarr_zip_store.parent.mkdir(exist_ok=True, parents=True)
        with zarr.storage.DirectoryStore(zarr_dir_store) as in_store, zarr.storage.ZipStore(zarr_zip_store, mode="w") as out_store:
            zarr.convenience.copy_store(in_store, out_store)
    print("Finished repacking")


if __name__ == "__main__":
    main()
