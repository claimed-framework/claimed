import json
import posixpath
import re
import argparse
from typing import Optional, Dict, List, Any, Tuple
from urllib.parse import quote, unquote
import boto3
from botocore.exceptions import ClientError

INDEX_SEPARATOR = "__i__"
KV_SEPARATOR = "="
FILENAME_SUFFIX = ".json"


def _encode_component(s: str) -> str:
    return quote(s, safe="")


def _decode_component(s: str) -> str:
    return unquote(s)


def _build_filename(key: str, indexes: Optional[Dict[str, str]] = None) -> str:
    parts = [_encode_component(key)]
    if indexes:
        for k in sorted(indexes.keys()):
            v = indexes[k]
            parts.append(f"{_encode_component(k)}{KV_SEPARATOR}{_encode_component(str(v))}")
    return INDEX_SEPARATOR.join(parts) + FILENAME_SUFFIX


def _parse_filename(filename: str) -> Tuple[str, Dict[str, str]]:
    if not filename.endswith(FILENAME_SUFFIX):
        raise ValueError("invalid filename (missing .json suffix)")
    core = filename[:-len(FILENAME_SUFFIX)]
    parts = core.split(INDEX_SEPARATOR)
    if not parts:
        raise ValueError("invalid filename")
    key = _decode_component(parts[0])
    indexes: Dict[str, str] = {}
    for p in parts[1:]:
        if KV_SEPARATOR not in p:
            continue
        k_enc, v_enc = p.split(KV_SEPARATOR, 1)
        k = _decode_component(k_enc)
        v = _decode_component(v_enc)
        indexes[k] = v
    return key, indexes


class S3KVStore:
    def __init__(self, bucket: str, store_name: str, s3_client: Optional[Any] = None, endpoint_url: Optional[str] = None, aws_access_key_id: Optional[str] = None, aws_secret_access_key: Optional[str] = None):
        self.bucket = bucket
        self.store_name = store_name.strip("/")
        if s3_client is None:
            self.s3 = boto3.client(
                "s3",
                endpoint_url=endpoint_url,
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
            )
        else:
            self.s3 = s3_client

    def _prefix(self) -> str:
        return f"{self.store_name}/" if self.store_name else ""

    def _s3_key_for_filename(self, filename: str) -> str:
        return posixpath.join(self._prefix(), filename)

    def list(self, prefix: Optional[str] = None, max_keys: int = 1000) -> List[Dict[str, Any]]:
        s3_prefix = self._prefix()
        continuation_token = None
        results: List[Dict[str, Any]] = []

        while True:
            kwargs = {"Bucket": self.bucket, "Prefix": s3_prefix, "MaxKeys": max_keys}
            if continuation_token:
                kwargs["ContinuationToken"] = continuation_token
            resp = self.s3.list_objects_v2(**kwargs)
            contents = resp.get("Contents", [])
            for obj in contents:
                full_key = obj["Key"]
                filename = posixpath.basename(full_key)
                try:
                    logical_key, indexes = _parse_filename(filename)
                except ValueError:
                    continue
                if prefix and not logical_key.startswith(prefix):
                    continue
                results.append({
                    "s3_key": full_key,
                    "filename": filename,
                    "key": logical_key,
                    "indexes": indexes,
                    "size": obj.get("Size", 0),
                    "last_modified": obj.get("LastModified"),
                })
            if not resp.get("IsTruncated"):
                break
            continuation_token = resp.get("NextContinuationToken")

        return results

    def _match_indexes(self, item_indexes: Dict[str, str], filt: Dict[str, Any]) -> bool:
        for fk, fv in filt.items():
            if fk not in item_indexes:
                return False
            val = item_indexes[fk]
            if isinstance(fv, (list, tuple, set)):
                if val not in {str(x) for x in fv}:
                    return False
            elif isinstance(fv, re.Pattern):
                if not fv.search(val):
                    return False
            else:
                if val != str(fv):
                    return False
        return True

    def get(self, key: str, index_filter: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        matches = self._find_objects_for_key(key, index_filter=index_filter)
        if not matches:
            raise KeyError(f"key not found: {key} (filter={index_filter})")
        if len(matches) > 1:
            raise ValueError(f"multiple objects match key={key}; refine using index_filter: {matches}")
        s3_key = matches[0]["s3_key"]
        try:
            resp = self.s3.get_object(Bucket=self.bucket, Key=s3_key)
            body = resp["Body"].read()
            return json.loads(body.decode("utf-8"))
        except ClientError as e:
            raise IOError(f"s3 get_object failed: {e}")

    def put(self, key: str, value: Dict[str, Any], indexes: Optional[Dict[str, Any]] = None, overwrite: bool = False) -> str:
        if overwrite:
            existing = self._find_objects_for_key(key)
            for obj in existing:
                self.s3.delete_object(Bucket=self.bucket, Key=obj["s3_key"])

        filename = _build_filename(key, {k: str(v) for k, v in (indexes or {}).items()})
        s3_key = self._s3_key_for_filename(filename)
        if not overwrite:
            try:
                self.s3.head_object(Bucket=self.bucket, Key=s3_key)
                raise FileExistsError(f"object already exists: {s3_key}")
            except ClientError as e:
                if e.response["Error"]["Code"] not in ("404", "NotFound", "NoSuchKey"):
                    raise

        payload = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.s3.put_object(Bucket=self.bucket, Key=s3_key, Body=payload, ContentType="application/json")
        return s3_key

    def update(self, key: str, value: Dict[str, Any], index_filter: Optional[Dict[str, Any]] = None, new_indexes: Optional[Dict[str, Any]] = None) -> str:
        matches = self._find_objects_for_key(key, index_filter=index_filter)
        if not matches:
            raise KeyError(f"no object matches key={key} index_filter={index_filter}")
        if len(matches) > 1:
            raise ValueError(f"multiple objects match key={key} index_filter={index_filter}: {matches}")

        old = matches[0]
        target_indexes = new_indexes if new_indexes is not None else old["indexes"]
        new_filename = _build_filename(key, {k: str(v) for k, v in (target_indexes or {}).items()})
        new_s3_key = self._s3_key_for_filename(new_filename)
        payload = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.s3.put_object(Bucket=self.bucket, Key=new_s3_key, Body=payload, ContentType="application/json")
        if old["s3_key"] != new_s3_key:
            self.s3.delete_object(Bucket=self.bucket, Key=old["s3_key"])
        return new_s3_key

    def delete(self, key: str, index_filter: Optional[Dict[str, Any]] = None) -> int:
        matches = self._find_objects_for_key(key, index_filter=index_filter)
        count = 0
        for obj in matches:
            self.s3.delete_object(Bucket=self.bucket, Key=obj["s3_key"])
            count += 1
        return count

    def search(self, index_filter: Dict[str, Any]) -> List[Dict[str, Any]]:
        all_items = self.list()
        return [it for it in all_items if self._match_indexes(it["indexes"], index_filter)]

    def _find_objects_for_key(self, key: str, index_filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        candidates = self.list(prefix=key)
        if index_filter is None:
            return candidates
        return [c for c in candidates if self._match_indexes(c["indexes"], index_filter)]


# ---------------- CLI ----------------
def main():
    parser = argparse.ArgumentParser(description="S3 KV Store CLI")
    parser.add_argument("bucket")
    parser.add_argument("store")
    parser.add_argument("--endpoint")

    sub = parser.add_subparsers(dest="cmd", required=True)

    # put
    sp = sub.add_parser("put")
    sp.add_argument("key")
    sp.add_argument("--indexes", type=json.loads, default="{}")
    sp.add_argument("--value")
    sp.add_argument("--value-file")
    sp.add_argument("--overwrite", action="store_true")

    # get
    sp = sub.add_parser("get")
    sp.add_argument("key")
    sp.add_argument("--filter", type=json.loads, default="{}")

    # update
    sp = sub.add_parser("update")
    sp.add_argument("key")
    sp.add_argument("--filter", type=json.loads, default="{}")
    sp.add_argument("--new-indexes", type=json.loads, default=None)
    sp.add_argument("--value")
    sp.add_argument("--value-file")

    # delete
    sp = sub.add_parser("delete")
    sp.add_argument("key")
    sp.add_argument("--filter", type=json.loads, default="{}")

    # list
    sp = sub.add_parser("list")
    sp.add_argument("--prefix")

    # search
    sp = sub.add_parser("search")
    sp.add_argument("--filter", type=json.loads, required=True)

    args = parser.parse_args()
    store = MLX(bucket=args.bucket, store_name=args.store, endpoint_url=args.endpoint)

    if args.cmd == "put":
        if args.value_file:
            value = json.load(open(args.value_file))
        else:
            value = json.loads(args.value)
        key = store.put(args.key, value, indexes=args.indexes, overwrite=args.overwrite)
        print(key)

    elif args.cmd == "get":
        value = store.get(args.key, index_filter=args.filter)
        print(json.dumps(value, indent=2))

    elif args.cmd == "update":
        if args.value_file:
            value = json.load(open(args.value_file))
        else:
            value = json.loads(args.value)
        key = store.update(args.key, value, index_filter=args.filter, new_indexes=args.new_indexes)
        print(key)

    elif args.cmd == "delete":
        count = store.delete(args.key, index_filter=args.filter)
        print(f"Deleted {count} object(s)")

    elif args.cmd == "list":
        items = store.list(prefix=args.prefix)
        print(json.dumps(items, indent=2, default=str))

    elif args.cmd == "search":
        items = store.search(args.filter)
        print(json.dumps(items, indent=2, default=str))


if __name__ == "__main__":
    main()
