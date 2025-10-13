from sqlalchemy import select
from sqlalchemy.orm import Session
import pandas as pd
from src.db.models import Message, Media, Channel


MANDATORY = [
    "channel_id","message_id","date","is_fwd","fwd_src_channel_id","mime","size","duration_s","width","height","s3_path","fpv_confidence", "source_url"
]


def export_manifest(db: Session, out_path: str):
    q = (
        select(
        Message.channel_id, Message.message_id, Message.date, Message.is_fwd, Message.fwd_src_channel_id,
        Media.mime, Media.size, Media.duration_s, Media.width, Media.height, Media.s3_path, Media.fpv_confidence,
        Channel.username
        )
        .join(Media, Media.message_pk == Message.id, isouter=True)
        .join(Channel, Channel.id == Message.channel_id, isouter=True)
    )
    rows = db.execute(q).all()
    cols = [*MANDATORY[:-1], "username"]  # додамо username для посилань
    df = pd.DataFrame(rows, columns=cols)

    def make_link(row):
        if row["username"]:
            return f"https://t.me/{row['username']}/{row['message_id']}"
        cid = abs(int(row["channel_id"]))
        if cid > 10**12:
            cid = cid - 1000000000000
        return f"https://t.me/c/{cid}/{row['message_id']}"

    df["source_url"] = df.apply(make_link, axis=1)
    df = df.drop(columns=["username"])

    if out_path.endswith(".parquet"):
        df.to_parquet(out_path, index=False)
    else:
        df.to_csv(out_path, index=False)
    return out_path
