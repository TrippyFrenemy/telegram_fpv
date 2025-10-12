from sqlalchemy import text
from sqlalchemy.orm import Session

def daily_stats(db: Session):
    sql = text(
        """
        SELECT date_trunc('day', date) AS day,
            COUNT(*) FILTER (WHERE has_media) AS msgs_with_media,
            COUNT(*) FILTER (WHERE NOT has_media) AS msgs_no_media,
            COUNT(media.sha256) AS videos
        FROM messages
        LEFT JOIN media ON media.message_pk = messages.id
        GROUP BY 1
        ORDER BY 1 DESC
        LIMIT 30
        """
    )
    return db.execute(sql).all()
