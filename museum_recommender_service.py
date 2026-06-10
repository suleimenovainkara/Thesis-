import os
import shutil
import tempfile
import hashlib
import hmac
import secrets
from typing import Optional, List, Dict, Any

import numpy as np 
import pandas as pd
import psycopg2
from PIL import Image, UnidentifiedImageError
import torch
import clip


DB_CONFIG = {
    "dbname":   "museumdb",
    "user":     "postgres",
    "password": "1234",
    "host":     "localhost",
    "port":     "5432"
}


class MuseumRecommenderService:
    def __init__(self, embeddings_path: str, metadata_path: str, model_name: str = "ViT-B/32"):
        print("Loading data...")

        self.embeddings = np.load(embeddings_path).astype(np.float32)
        self.metadata   = pd.read_csv(metadata_path)

        self.metadata.columns = [c.strip().lower() for c in self.metadata.columns]

        if "filenames" in self.metadata.columns and "filename" not in self.metadata.columns:
            self.metadata.rename(columns={"filenames": "filename"}, inplace=True)

        for col in ["filename", "artist", "style"]:
            if col not in self.metadata.columns:
                raise ValueError(f"Column '{col}' not found in metadata")

        self.metadata["filename"]   = self.metadata["filename"].fillna("").astype(str)
        self.metadata["artist"]     = self.metadata["artist"].fillna("Unknown").astype(str)
        self.metadata["style"]      = self.metadata["style"].fillna("Unknown").astype(str)
        self.metadata["image_path"] = self.metadata.get("image_path", pd.Series("")).fillna("").astype(str)

        if len(self.embeddings) != len(self.metadata):
            raise ValueError(
                f"Embeddings ({len(self.embeddings)}) != metadata rows ({len(self.metadata)})"
            )

        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        self.embeddings = self.embeddings / norms  

        self.filename_to_index: Dict[str, int] = {
            row["filename"]: idx
            for idx, row in self.metadata.iterrows()
            if row["filename"].strip()
        }


        print("Caching artwork IDs from database...")
        self._filename_to_artwork_id: Dict[str, int] = self._load_filename_to_artwork_id()
        print(f"  Cached {len(self._filename_to_artwork_id)} artwork IDs")

        # CLIP модель 
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Device:", self.device.upper())
        self.model, self.preprocess = clip.load(model_name, device=self.device)

        print(f"Embeddings: {self.embeddings.shape}")
        print(f"Metadata:   {self.metadata.shape}")

    # DB HELPERS

    def get_connection(self):
        return psycopg2.connect(**DB_CONFIG)

    def execute_query(self, query: str, params=None):
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params or ())
            conn.commit()

    def fetch_all(self, query: str, params=None) -> List[Dict]:
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params or ())
                cols = [d[0] for d in cur.description]
                return [dict(zip(cols, row)) for row in cur.fetchall()]

    def fetch_one(self, query: str, params=None) -> Optional[Dict]:
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params or ())
                row = cur.fetchone()
                if row is None:
                    return None
                cols = [d[0] for d in cur.description]
                return dict(zip(cols, row))

    def _user_exists(self, user_id: int) -> bool:
        return self.fetch_one("SELECT 1 AS ok FROM users WHERE id = %s;", (user_id,)) is not None

    def _artwork_exists(self, artwork_id: int) -> bool:
        return self.fetch_one("SELECT 1 AS ok FROM artworks WHERE id = %s;", (artwork_id,)) is not None

    def _load_filename_to_artwork_id(self) -> Dict[str, int]:
        """
        Загружает ВСЕ filename → id из таблицы artworks ОДИН РАЗ при старте.
        Дальше поиск O(1) без запросов в БД.
        """
        rows = self.fetch_all("SELECT id, filename FROM artworks WHERE filename IS NOT NULL;")
        return {row["filename"].strip(): row["id"] for row in rows if row["filename"].strip()}

    def get_artwork_id_by_filename(self, filename: str) -> Optional[int]:
        """
        Теперь O(1) — смотрим в кэш, а не в БД.
        """
        return self._filename_to_artwork_id.get((filename or "").strip())

    # AUTH

    def _validate_username(self, username: str) -> str:
        username = (username or "").strip()
        if not username:            raise ValueError("Username cannot be empty.")
        if len(username) < 3:       raise ValueError("Username must contain at least 3 characters.")
        if len(username) > 50:      raise ValueError("Username is too long.")
        return username

    def _validate_password(self, password: str) -> str:
        password = password or ""
        if not password:            raise ValueError("Password cannot be empty.")
        if len(password) < 6:       raise ValueError("Password must contain at least 6 characters.")
        return password

    def _hash_password(self, password: str) -> str:
        salt = secrets.token_hex(16)
        h    = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100_000).hex()
        return f"{salt}${h}"

    def _verify_password(self, password: str, stored: str) -> bool:
        try:
            salt, expected = stored.split("$", 1)
        except ValueError:
            return False
        candidate = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100_000).hex()
        return hmac.compare_digest(candidate, expected)

    def create_user(self, username: str, password: str, role: str = "user") -> Dict[str, Any]:
        username = self._validate_username(username)
        password = self._validate_password(password)

        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT id FROM users WHERE username = %s;", (username,))
                if cur.fetchone():
                    raise ValueError("Username already exists.")

                cur.execute(
                    "INSERT INTO users (username, password_hash, role) "
                    "VALUES (%s, %s, %s) RETURNING id, username, role;",
                    (username, self._hash_password(password), role)
                )
                row = cur.fetchone()
            conn.commit()

        return {"user_id": row[0], "username": row[1], "role": row[2]}

    def authenticate_user(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        username = self._validate_username(username)
        self._validate_password(password)

        row = self.fetch_one(
            "SELECT id, username, password_hash, role FROM users WHERE username = %s;",
            (username,)
        )
        if not row or not self._verify_password(password, row["password_hash"]):
            return None

        return {"user_id": row["id"], "username": row["username"], "role": row["role"]}

    # =========================================================================
    # INTERACTIONS
    # =========================================================================

    def save_interaction(self, user_id: int, artwork_id: int, action_type: str) -> Dict[str, Any]:
        action_type = (action_type or "").strip().lower()
        if action_type not in {"view", "like", "skip"}:
            raise ValueError("action_type must be: view, like, skip")
        if not self._user_exists(user_id):    raise ValueError("User not found.")
        if not self._artwork_exists(artwork_id): raise ValueError("Artwork not found.")

        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT action_type FROM interactions
                    WHERE user_id = %s AND artwork_id = %s
                    ORDER BY created_at DESC LIMIT 1;
                """, (user_id, artwork_id))
                last = cur.fetchone()
                if last and last[0] == action_type:
                    return {"success": True, "message": "Already recorded."}

                cur.execute(
                    "INSERT INTO interactions (user_id, artwork_id, action_type) VALUES (%s, %s, %s);",
                    (user_id, artwork_id, action_type)
                )
            conn.commit()

        self.rebuild_and_save_user_profile(user_id)
        return {"success": True}

    def get_liked_filenames_from_db(self, user_id: int) -> List[str]:
        rows = self.fetch_all("""
            SELECT DISTINCT a.filename FROM interactions i
            JOIN artworks a ON i.artwork_id = a.id
            WHERE i.user_id = %s AND i.action_type = 'like';
        """, (user_id,))
        return [r["filename"] for r in rows]

    def get_seen_filenames_from_db(self, user_id: int) -> List[str]:
        rows = self.fetch_all("""
            SELECT DISTINCT a.filename FROM interactions i
            JOIN artworks a ON i.artwork_id = a.id
            WHERE i.user_id = %s AND i.action_type IN ('view','like','skip');
        """, (user_id,))
        return [r["filename"] for r in rows]

    def get_user_interactions(self, user_id: int) -> List[Dict]:
        return self.fetch_all("""
            SELECT a.filename, i.action_type, i.created_at
            FROM interactions i JOIN artworks a ON i.artwork_id = a.id
            WHERE i.user_id = %s ORDER BY i.created_at ASC;
        """, (user_id,))

    def get_artwork_id_by_filename_db(self, filename: str) -> Optional[int]:
        """Прямой запрос в БД — используй только если нужна гарантия свежих данных"""
        row = self.fetch_one("SELECT id FROM artworks WHERE filename = %s;", (filename,))
        return row["id"] if row else None

    # FAVORITES

    def add_to_favorites(self, user_id: int, artwork_id: int) -> Dict[str, Any]:
        if not self._user_exists(user_id):      raise ValueError("User not found.")
        if not self._artwork_exists(artwork_id): raise ValueError("Artwork not found.")

        self.execute_query("""
            INSERT INTO favorites (user_id, artwork_id)
            VALUES (%s, %s) ON CONFLICT (user_id, artwork_id) DO NOTHING;
        """, (user_id, artwork_id))

        self.rebuild_and_save_user_profile(user_id)
        return {"status": "added"}

    def remove_from_favorites(self, user_id: int, artwork_id: int) -> Dict[str, Any]:
        if not self._user_exists(user_id): raise ValueError("User not found.")

        self.execute_query(
            "DELETE FROM favorites WHERE user_id = %s AND artwork_id = %s;",
            (user_id, artwork_id)
        )
        self.rebuild_and_save_user_profile(user_id)
        return {"status": "removed"}

    def get_favorites(self, user_id: int) -> List[Dict]:
        if not self._user_exists(user_id): raise ValueError("User not found.")
        return self.fetch_all("""
            SELECT a.id, a.filename, a.artist, a.style, a.image_path, f.created_at AS favorited_at
            FROM favorites f JOIN artworks a ON f.artwork_id = a.id
            WHERE f.user_id = %s ORDER BY f.created_at DESC;
        """, (user_id,))

    def get_favorite_filenames_from_db(self, user_id: int) -> List[str]:
        rows = self.fetch_all("""
            SELECT DISTINCT a.filename FROM favorites f
            JOIN artworks a ON f.artwork_id = a.id WHERE f.user_id = %s;
        """, (user_id,))
        return [r["filename"] for r in rows]

    # PROFILE BUILDING

    def filenames_to_indices(self, filenames: List[str]) -> List[int]:
        return [self.filename_to_index[f] for f in filenames if f in self.filename_to_index]

    def build_user_profile_basic(self, liked_indices: List[int]) -> Optional[np.ndarray]:
        if not liked_indices:
            return None
        vec  = np.mean(self.embeddings[liked_indices], axis=0)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else None

    def build_user_profile_weighted_from_events(
        self, user_id: int, action_weights=None
    ) -> Optional[np.ndarray]:
        if action_weights is None:
            action_weights = {"view": 0.15, "like": 1.0, "skip": -0.35}

        interactions = self.get_user_interactions(user_id)
        if not interactions:
            return None

        vecs, ws = [], []
        for row in interactions:
            idx = self.filename_to_index.get(row["filename"])
            w   = action_weights.get(row["action_type"], 0.0)
            if idx is not None and w != 0.0:
                vecs.append(self.embeddings[idx])
                ws.append(w)

        if not vecs:
            return None

        vec  = np.sum(np.array(vecs) * np.array(ws).reshape(-1, 1), axis=0)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else None

    def build_user_profile_weighted_combined(
        self, user_id: int, favorite_weight: float = 2.0, action_weights=None
    ) -> Optional[np.ndarray]:
        iv = self.build_user_profile_weighted_from_events(user_id, action_weights)

        fav_idx = self.filenames_to_indices(self.get_favorite_filenames_from_db(user_id))
        fv = None
        if fav_idx:
            v    = np.mean(self.embeddings[fav_idx], axis=0)
            norm = np.linalg.norm(v)
            fv   = v / norm if norm > 0 else None

        if iv is None and fv is None:
            return None

        combined = (iv if iv is not None else 0) + favorite_weight * (fv if fv is not None else 0)
        norm = np.linalg.norm(combined)
        return combined / norm if norm > 0 else None

    def save_global_interest_vector(self, user_id: int, vec: np.ndarray):
        if vec is None:
            return
        self.execute_query(
            "UPDATE users SET global_interest_vector = %s WHERE id = %s;",
            (",".join(map(str, vec.tolist())), user_id)
        )

    def load_global_interest_vector(self, user_id: int) -> Optional[np.ndarray]:
        row = self.fetch_one(
            "SELECT global_interest_vector FROM users WHERE id = %s;", (user_id,)
        )
        if not row or not row["global_interest_vector"]:
            return None
        vec  = np.array([float(x) for x in row["global_interest_vector"].split(",")], dtype=np.float32)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else None

    def rebuild_and_save_user_profile(
        self, user_id: int, favorite_weight: float = 2.0, action_weights=None
    ) -> Optional[np.ndarray]:
        if not self._user_exists(user_id):
            return None
        vec = self.build_user_profile_weighted_combined(user_id, favorite_weight, action_weights)
        if vec is not None:
            self.save_global_interest_vector(user_id, vec)
        return vec

    def get_preferred_styles(self, user_id: int) -> Dict[str, float]:
        liked_idx = self.filenames_to_indices(self.get_liked_filenames_from_db(user_id))
        fav_idx   = self.filenames_to_indices(self.get_favorite_filenames_from_db(user_id))

        scores: Dict[str, float] = {}
        for idx in liked_idx:
            s = self.metadata.iloc[idx]["style"]
            scores[s] = scores.get(s, 0.0) + 1.0
        for idx in fav_idx:
            s = self.metadata.iloc[idx]["style"]
            scores[s] = scores.get(s, 0.0) + 2.0

        total = sum(scores.values())
        return {s: v / total for s, v in scores.items()} if total > 0 else {}

    # REASON BUILDER

    def _build_reason(self, row, preferred_styles=None, query_style=None) -> str:
        style  = row["style"]
        artist = row["artist"]
        if query_style and style == query_style:
            return f"Similar visual characteristics and the same style: {style}."
        if preferred_styles and style in preferred_styles:
            return f"Matches one of your preferred styles: {style}."
        return f"Recommended based on visual similarity, artist: {artist}."

    # DIVERSITY FILTER

    def _apply_diversity_filters(
        self, candidates: List[Dict], top_n: int = 5,
        max_per_style: int = 2, max_per_artist: int = 1
    ) -> List[Dict]:
        results, seen, style_c, artist_c = [], set(), {}, {}
        for item in candidates:
            fn     = item["filename"]
            style  = item["style"]
            artist = item["artist"]
            if not fn.strip() or fn in seen:               continue
            if style_c.get(style, 0)   >= max_per_style:   continue
            if artist_c.get(artist, 0) >= max_per_artist:  continue

            results.append(item)
            seen.add(fn)
            style_c[style]   = style_c.get(style, 0) + 1
            artist_c[artist] = artist_c.get(artist, 0) + 1
            if len(results) == top_n:
                break
        return results

    # ONBOARDING

    def get_onboarding_candidates(self, total_n: int = 10, per_style: int = 1, random_state: int = 42) -> List[Dict]:
        if total_n <= 0:
            return []

        df  = self.metadata[["filename", "artist", "style", "image_path"]].copy()
        df  = df.reset_index().rename(columns={"index": "embedding_index"})
        df  = df[df["filename"].str.strip() != ""]
        rng = np.random.default_rng(random_state)

        grouped = []
        styles  = df["style"].dropna().unique().tolist()
        rng.shuffle(styles)

        for style in styles:
            subset = df[df["style"] == style]
            if subset.empty:
                continue
            n   = min(per_style, len(subset))
            idx = rng.choice(subset.index.to_numpy(), size=n, replace=False)
            grouped.append(subset.loc[idx])

        if not grouped:
            return []

        out = pd.concat(grouped, ignore_index=True)
        out = out.sample(frac=1, random_state=random_state).reset_index(drop=True).head(total_n)
        return out[["embedding_index", "filename", "artist", "style", "image_path"]].to_dict(orient="records")

    def submit_onboarding(
        self, user_id: int, selected_embedding_indices: List[int], top_n: int = 5
    ) -> List[Dict]:
        if not self._user_exists(user_id):
            raise ValueError("User not found.")

        for emb_idx in sorted(set(selected_embedding_indices)):
            if not isinstance(emb_idx, int) or not (0 <= emb_idx < len(self.metadata)):
                continue
            filename   = self.metadata.iloc[emb_idx]["filename"]
            artwork_id = self.get_artwork_id_by_filename(filename)
            if artwork_id is None:
                continue
            self.save_interaction(user_id, artwork_id, "like")

        self.rebuild_and_save_user_profile(user_id)
        return self.recommend_for_user(user_id=user_id, top_n=top_n, rebuild_profile=False)

    # IMAGE EMBEDDING

    def extract_embedding_from_image(self, image_path: str) -> np.ndarray:
        with Image.open(image_path) as img:
            tensor = self.preprocess(img.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.model.encode_image(tensor)
            emb /= emb.norm(dim=-1, keepdim=True)
        return emb.cpu().numpy().flatten().astype(np.float32)

    # RECOMMEND BY IMAGE

    def recommend_by_image(
        self, image_path: str, top_n: int = 5,
        max_per_style: int = 2, max_per_artist: int = 1
    ) -> List[Dict]:
        if not os.path.exists(image_path):
            raise ValueError("Image file not found.")

        q_vec = self.extract_embedding_from_image(image_path)          
        sims  = self.embeddings @ q_vec                                  # dot = cosine

        candidates = []
        for idx in np.argsort(sims)[::-1]:
            row = self.metadata.iloc[idx]
            candidates.append({
                "artwork_id":       self.get_artwork_id_by_filename(row["filename"]),
                "embedding_index":  int(idx),
                "filename":         row["filename"],
                "artist":           row["artist"],
                "style":            row["style"],
                "image_path":       row["image_path"],
                "similarity":       float(sims[idx]),
                "reason":           self._build_reason(row),
            })

        return self._apply_diversity_filters(candidates, top_n, max_per_style, max_per_artist)

    # RECOMMEND BY ARTWORK

    def recommend_by_artwork(
        self, filename: str, top_n: int = 5,
        style_bonus_value: float = 0.02,
        max_per_style: int = 2, max_per_artist: int = 1
    ) -> List[Dict]:
        filename  = (filename or "").strip()
        query_idx = self.filename_to_index.get(filename)
        if query_idx is None:
            return []

        q_vec       = self.embeddings[query_idx]        
        sims        = self.embeddings @ q_vec           # dot = cosine
        query_style = self.metadata.iloc[query_idx]["style"]

        candidates = []
        for idx, sim in enumerate(sims):
            if idx == query_idx:
                continue
            row         = self.metadata.iloc[idx]
            style_bonus = style_bonus_value if row["style"] == query_style else 0.0
            candidates.append({
                "artwork_id":       self.get_artwork_id_by_filename(row["filename"]),
                "embedding_index":  int(idx),
                "filename":         row["filename"],
                "artist":           row["artist"],
                "style":            row["style"],
                "image_path":       row["image_path"],
                "similarity":       float(sim),
                "style_bonus":      float(style_bonus),
                "final_score":      float(sim) + style_bonus,
                "reason":           self._build_reason(row, query_style=query_style),
            })

        candidates.sort(key=lambda x: x["final_score"], reverse=True)
        return self._apply_diversity_filters(candidates, top_n, max_per_style, max_per_artist)


    # RECOMMEND FOR USER  

    def recommend_for_user(
        self,
        user_id:              int,
        top_n:                int   = 5,
        use_weighted_profile: bool  = True,
        style_bonus_value:    float = 0.02,
        max_per_style:        int   = 2,
        max_per_artist:       int   = 1,
        rebuild_profile:      bool  = True,
    ) -> List[Dict]:
        if not self._user_exists(user_id):
            raise ValueError("User not found.")

        if use_weighted_profile:
            if rebuild_profile:
                user_vec = self.rebuild_and_save_user_profile(user_id)
            else:
                user_vec = self.load_global_interest_vector(user_id)
                if user_vec is None:
                    user_vec = self.rebuild_and_save_user_profile(user_id)
        else:
            liked_idx   = self.filenames_to_indices(self.get_liked_filenames_from_db(user_id))
            fav_idx     = self.filenames_to_indices(self.get_favorite_filenames_from_db(user_id))
            all_idx     = sorted(set(liked_idx) | set(fav_idx))
            user_vec    = self.build_user_profile_basic(all_idx)

        if user_vec is None:
            return []

        sims = self.embeddings @ user_vec.flatten().astype(np.float32)

        preferred_styles = self.get_preferred_styles(user_id)

        seen = set(self.get_seen_filenames_from_db(user_id)) | \
               set(self.get_favorite_filenames_from_db(user_id))

        candidates = []
        for idx in np.argsort(sims)[::-1]:
            row      = self.metadata.iloc[idx]
            filename = row["filename"]

            if not filename.strip() or filename in seen:
                continue

            sim         = float(sims[idx])
            style_bonus = preferred_styles.get(row["style"], 0.0) * style_bonus_value

            candidates.append({
                "artwork_id":       self._filename_to_artwork_id.get(filename),  # кэш!
                "embedding_index":  int(idx),
                "filename":         filename,
                "artist":           row["artist"],
                "style":            row["style"],
                "image_path":       row["image_path"],
                "similarity":       sim,
                "style_bonus":      float(style_bonus),
                "final_score":      sim + style_bonus,
                "reason":           self._build_reason(row, preferred_styles=preferred_styles),
            })

      
            if len(candidates) >= top_n * 10:
                break

        candidates.sort(key=lambda x: x["final_score"], reverse=True)
        return self._apply_diversity_filters(candidates, top_n, max_per_style, max_per_artist)

    # UPLOAD HELPERS

    def recommend_by_uploaded_tempfile(self, upload_file_obj, top_n: int = 5) -> List[Dict]:
        if not upload_file_obj or not getattr(upload_file_obj, "filename", None):
            raise ValueError("No file was uploaded.")

        suffix = os.path.splitext(upload_file_obj.filename)[1].lower() or ".jpg"
        if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
            raise ValueError("Unsupported image format.")

        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                shutil.copyfileobj(upload_file_obj.file, tmp)
                temp_path = tmp.name

            try:
                with Image.open(temp_path) as img:
                    img.verify()
            except (UnidentifiedImageError, OSError):
                raise ValueError("Uploaded file is not a valid image.")

            return self.recommend_by_image(temp_path, top_n=top_n)

        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)