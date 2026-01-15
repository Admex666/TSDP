"""
Data import API endpoints
"""
from fastapi import APIRouter, Depends, File, UploadFile, Form, HTTPException
from sqlalchemy.orm import Session
from pathlib import Path
import shutil

from app.database import get_db
from app.schemas import ImportStatsResponse
from app.services import ImportService
from config import config

router = APIRouter(prefix="/import", tags=["import"])


@router.post("/stats", response_model=ImportStatsResponse)
async def import_stats(
    file: UploadFile = File(...),
    source: str = Form(...),
    auto_match_players: bool = Form(True),
    db: Session = Depends(get_db)
):
    """
    Import player statistics from CSV file
    
    Args:
        file: CSV file upload
        source: Data source (fbref, sofascore)
        auto_match_players: Automatically match players to database
    """
    # Validate source
    valid_sources = ["fbref", "sofascore"]
    if source.lower() not in valid_sources:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid source. Must be one of: {', '.join(valid_sources)}"
        )
    
    # Validate file extension
    if not file.filename.endswith('.csv'):
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are supported"
        )
    
    # Save uploaded file temporarily
    temp_file_path = config.IMPORTS_DIR / file.filename
    config.IMPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        with temp_file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save file: {str(e)}"
        )
    
    # Import data
    import_service = ImportService(db)
    result = import_service.import_csv(
        str(temp_file_path),
        source,
        auto_match_players
    )
    
    return ImportStatsResponse(**result)
