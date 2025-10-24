from datetime import datetime, timedelta
import os
import tempfile
import uuid
from pathlib import Path
import pytz

from src.config import settings
from src.store.storage_backend import get_storage_backend, StorageObject
from src.utils.video_splitter import split_video


STALE_DELTA = timedelta(minutes=2)


class VideoProcessor:
    """Handles video splitting and uploading"""
    
    def __init__(self):
        self.storage = get_storage_backend()
        self.temp_dir = tempfile.gettempdir()
    
    def process_video(self, src_key: str, skip_fresh: bool = False) -> bool:
        """
        Process single video: download, split, upload segments, delete original.
        
        Args:
            src_key: Source video key in storage
            skip_fresh: If True, skip videos modified recently
            
        Returns:
            True if processed successfully, False otherwise
        """
        # Check if video is too fresh
        if skip_fresh and self._is_too_fresh(src_key):
            print(f"[skip: too fresh] {src_key}")
            return False
        
        tmp_path = self._get_temp_path()
        
        try:
            # Download video
            self.storage.fget_object(src_key, tmp_path)
            
            # Split into segments
            segments = split_video(tmp_path, self.temp_dir, segment_s=5)
            
            # Upload segments
            seg_prefix = f"segments/{os.path.dirname(src_key)}"
            self._upload_segments(segments, seg_prefix)
            
            # Delete original video
            self.storage.delete(src_key)
            
            return True
            
        except Exception as e:
            print(f"[error] {src_key}: {e}")
            return False
            
        finally:
            # Cleanup temp file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    def _is_too_fresh(self, key: str) -> bool:
        """Check if video was modified too recently"""
        obj = self.storage.stat_object(key)
        if not obj or not obj.last_modified:
            return False
        
        now = datetime.now(pytz.timezone("Europe/Kyiv"))
        return (now - obj.last_modified) < STALE_DELTA
    
    def _get_temp_path(self) -> str:
        """Generate unique temp file path"""
        return os.path.join(self.temp_dir, f"{uuid.uuid4()}.mp4")
    
    def _upload_segments(self, segments: list[str], prefix: str) -> None:
        """Upload segment files to storage"""
        for seg_path in segments:
            seg_name = os.path.basename(seg_path)
            dst_key = f"{prefix}/{seg_name}"
            
            self.storage.fput_object(dst_key, seg_path)
            os.remove(seg_path)
            
            print(f"[uploaded] {dst_key}")


def process_s3(prefix: str = "telegram"):
    """
    Process all videos in telegram/ prefix.
    Splits each video into 5-second segments and uploads them.
    """
    processor = VideoProcessor()
    storage = get_storage_backend()
    
    print("Scanning telegram/ for videos to process...")
    
    processed = 0
    skipped = 0
    
    for obj in storage.list_objects(prefix=f"{prefix}/", recursive=True):
        if processor.process_video(obj.object_name, skip_fresh=True):
            processed += 1
        else:
            skipped += 1
    
    print(f"\nProcessing complete:")
    print(f"  Processed: {processed}")
    print(f"  Skipped: {skipped}")


if __name__ == "__main__":
    process_s3()
