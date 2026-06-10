from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from museum_recommender_service import MuseumRecommenderService
import traceback

EMBEDDINGS_PATH = r"D:\final_project\embeddings_balanced.npy"
METADATA_PATH = r"D:\final_project\metadata_valid_balanced.csv"

app = FastAPI(title="Museum Recommender API")

service = MuseumRecommenderService(
    embeddings_path=EMBEDDINGS_PATH,
    metadata_path=METADATA_PATH
)


# Pydantic schemas
class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6, max_length=128)
    role: Optional[str] = "user"

    @field_validator("username")
    @classmethod
    def validate_username(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Username cannot be empty.")
        return value


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=1, max_length=128)

    @field_validator("username")
    @classmethod
    def validate_username(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Username cannot be empty.")
        return value


class InteractionRequest(BaseModel):
    user_id: int
    artwork_id: int
    action_type: str

    @field_validator("action_type")
    @classmethod
    def validate_action_type(cls, value: str) -> str:
        value = value.strip().lower()
        allowed = {"view", "like", "skip"}
        if value not in allowed:
            raise ValueError("action_type must be one of: view, like, skip")
        return value


class FavoriteRequest(BaseModel):
    user_id: int
    artwork_id: int


class OnboardingSubmitRequest(BaseModel):
    user_id: int
    selected_embedding_indices: List[int]
    top_n: Optional[int] = 5


class UserRecommendRequest(BaseModel):
    user_id: int
    top_n: Optional[int] = 5
    use_weighted_profile: Optional[bool] = True
    style_bonus_value: Optional[float] = 0.02
    max_per_style: Optional[int] = 2
    max_per_artist: Optional[int] = 1
    rebuild_profile: Optional[bool] = True


class ArtworkRecommendRequest(BaseModel):
    filename: str
    top_n: Optional[int] = 5
    style_bonus_value: Optional[float] = 0.02
    max_per_style: Optional[int] = 2
    max_per_artist: Optional[int] = 1

    @field_validator("filename")
    @classmethod
    def validate_filename(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Filename cannot be empty.")
        return value


# =====================================================
# Routes
# =====================================================
@app.get("/")
def root():
    return {"message": "Museum Recommender API is running"}


@app.post("/users/register", status_code=status.HTTP_201_CREATED)
def register_user(payload: RegisterRequest):
    try:
        result = service.create_user(
            username=payload.username,
            password=payload.password,
            role=payload.role
        )
        return result

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Registration failed: {str(e)}"
        )


@app.post("/users/login")
def login_user(payload: LoginRequest):
    try:
        user = service.authenticate_user(
            username=payload.username,
            password=payload.password
        )

        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password."
            )

        return user

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Login failed: {str(e)}"
        )


@app.get("/onboarding/candidates")
def get_onboarding_candidates(total_n: int = 10, per_style: int = 1, random_state: int = 42):
    try:
        candidates = service.get_onboarding_candidates(
            total_n=total_n,
            per_style=per_style,
            random_state=random_state
        )
        return {"candidates": candidates}

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not load onboarding candidates: {str(e)}"
        )


@app.post("/onboarding/submit")
def submit_onboarding(payload: OnboardingSubmitRequest):
    try:
        if not payload.selected_embedding_indices:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="At least one artwork must be selected."
            )

        recommendations = service.submit_onboarding(
            user_id=payload.user_id,
            selected_embedding_indices=payload.selected_embedding_indices,
            top_n=payload.top_n
        )
        return {"recommendations": recommendations}

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Onboarding failed: {str(e)}"
        )


@app.post("/recommend/for-user")
def recommend_for_user(payload: UserRecommendRequest):
    try:
        recommendations = service.recommend_for_user(
            user_id=payload.user_id,
            top_n=payload.top_n,
            use_weighted_profile=payload.use_weighted_profile,
            style_bonus_value=payload.style_bonus_value,
            max_per_style=payload.max_per_style,
            max_per_artist=payload.max_per_artist,
            rebuild_profile=payload.rebuild_profile
        )
        return {"recommendations": recommendations}

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
      print("ERROR in /recommend/for-user")
      traceback.print_exc()
      raise HTTPException(
         status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
         detail=f"Recommendation failed: {str(e)}"
    )


@app.post("/recommend/by-artwork")
def recommend_by_artwork(payload: ArtworkRecommendRequest):
    try:
        recommendations = service.recommend_by_artwork(
            filename=payload.filename,
            top_n=payload.top_n,
            style_bonus_value=payload.style_bonus_value,
            max_per_style=payload.max_per_style,
            max_per_artist=payload.max_per_artist
        )
        return {"recommendations": recommendations}

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Artwork recommendation failed: {str(e)}"
        )


@app.post("/recommend/by-image")
def recommend_by_image(file: UploadFile = File(...), top_n: int = 5):
    try:
        recommendations = service.recommend_by_uploaded_tempfile(file, top_n=top_n)
        return {"recommendations": recommendations}

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Image recommendation failed: {str(e)}"
        )


@app.post("/interactions")
def save_interaction(payload: InteractionRequest):
    try:
        result = service.save_interaction(
            user_id=payload.user_id,
            artwork_id=payload.artwork_id,
            action_type=payload.action_type
        )
        return result

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not save interaction: {str(e)}"
        )


@app.post("/favorites/add")
def add_to_favorites(payload: FavoriteRequest):
    try:
        result = service.add_to_favorites(
            user_id=payload.user_id,
            artwork_id=payload.artwork_id
        )
        return result

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not add to favorites: {str(e)}"
        )


@app.post("/favorites/remove")
def remove_from_favorites(payload: FavoriteRequest):
    try:
        result = service.remove_from_favorites(
            user_id=payload.user_id,
            artwork_id=payload.artwork_id
        )
        return result

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not remove from favorites: {str(e)}"
        )


@app.get("/favorites/{user_id}")
def get_favorites(user_id: int):
    try:
        favorites = service.get_favorites(user_id=user_id)
        return {"favorites": favorites}

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not fetch favorites: {str(e)}"
        )