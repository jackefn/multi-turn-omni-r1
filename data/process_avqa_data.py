import os
import pandas as pd
from tqdm import tqdm
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

csv_path = "data/AVQA/AVQA/AVQA_dataset/avqa_download_urls.csv"
output_dir = "data/AVQA/AVQA/AVQA_dataset/raw_videos/"
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(csv_path)

failed_log = "data/AVQA/AVQA/AVQA_dataset/failed_downloads.txt"


def download_video(yt_id):
    
    yt_id = str(yt_id).strip()

    outpath = os.path.join(output_dir, f"{yt_id}.mp4")
    url = f"https://www.youtube.com/watch?v={yt_id}"

    
    if os.path.exists(outpath):
        return (yt_id, "exists")

    cmd = [
        "yt-dlp",
        "-f", "bestvideo+bestaudio/best",
        "-o", outpath,
        url
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return (yt_id, "success")
    except Exception as e:
        return (yt_id, f"fail: {e}")


if __name__ == "__main__":
    yt_ids = df["youtube ID"].tolist()

    
    max_workers = max(1, multiprocessing.cpu_count() - 1)
    print(f"Using {max_workers} parallel workers.")

    failed_ids = []

   
    with ProcessPoolExecutor(max_workers=80) as executor:
        futures = {executor.submit(download_video, vid): vid for vid in yt_ids}

        for future in tqdm(as_completed(futures), total=len(futures)):
            yt_id, status = future.result()

            if status == "success":
                pass
            elif status == "exists":
                pass
            else:
                
                failed_ids.append(yt_id)
                print(f"[Fail] {yt_id} => {status}")

    
    if failed_ids:
        with open(failed_log, "w") as f:
            for vid in failed_ids:
                f.write(vid + "\n")

    print(f"\nFinished. Failed: {len(failed_ids)} videos.")
    print(f"Failed list saved to: {failed_log}")
