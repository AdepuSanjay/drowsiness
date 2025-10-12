from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

# Create FastAPI instance
app = FastAPI(title="Simple API", version="1.0.0")

# Pydantic model for data validation
class Item(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    price: float

# In-memory database (for demo purposes)
fake_db = []

# Root endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to Simple FastAPI!"}

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Get all items
@app.get("/items", response_model=List[Item])
async def get_all_items():
    return fake_db

# Get single item by ID
@app.get("/items/{item_id}", response_model=Item)
async def get_item(item_id: int):
    for item in fake_db:
        if item["id"] == item_id:
            return item
    raise HTTPException(status_code=404, detail="Item not found")

# Create new item
@app.post("/items", response_model=Item)
async def create_item(item: Item):
    # Check if item with same ID exists
    for existing_item in fake_db:
        if existing_item["id"] == item.id:
            raise HTTPException(status_code=400, detail="Item ID already exists")
    
    item_dict = item.dict()
    fake_db.append(item_dict)
    return item_dict

# Update item
@app.put("/items/{item_id}", response_model=Item)
async def update_item(item_id: int, item: Item):
    if item.id != item_id:
        raise HTTPException(status_code=400, detail="Item ID mismatch")
    
    for index, existing_item in enumerate(fake_db):
        if existing_item["id"] == item_id:
            fake_db[index] = item.dict()
            return fake_db[index]
    
    raise HTTPException(status_code=404, detail="Item not found")

# Delete item
@app.delete("/items/{item_id}")
async def delete_item(item_id: int):
    for index, item in enumerate(fake_db):
        if item["id"] == item_id:
            deleted_item = fake_db.pop(index)
            return {"message": "Item deleted", "deleted_item": deleted_item}
    
    raise HTTPException(status_code=404, detail="Item not found")

# Search items by name
@app.get("/items/search/")
async def search_items(name: str):
    results = [item for item in fake_db if name.lower() in item["name"].lower()]
    return {"results": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)