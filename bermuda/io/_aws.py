def get_awswrangler():
    try:
        import awswrangler as wr
    except ImportError as exc:
        raise ImportError(
            "S3 I/O requires the optional `awswrangler` dependency. "
            "Install `awswrangler` to use s3:// paths."
        ) from exc

    return wr
