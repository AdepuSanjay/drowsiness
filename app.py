from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form, WebSocket, WebSocketDisconnect, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import httpx
import json
import asyncio
import subprocess
import os
import shutil
import glob
from typing import Optional, Dict, Any, List
import time
import threading
from pathlib import Path
import tempfile
import uuid
import zipfile
import tarfile
from git import Repo
import git
import requests
from fastapi.staticfiles import StaticFiles
from datetime import datetime, timedelta
import hashlib
import jwt
from bson import ObjectId
import motor.motor_asyncio
from difflib import unified_diff

# MongoDB connection
MONGO_URI = "mongodb+srv://tkrcet:abc1234@cluster0.y4apc.mongodb.net/tkrcet"
client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)
db = client.tkrcet

# Collections
users_collection = db.users
sessions_collection = db.sessions
conversations_collection = db.conversations
file_diffs_collection = db.file_diffs

app = FastAPI(title="Cursor AI Clone", version="4.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()
SECRET_KEY = "DATASCIENCE"  # In production, use environment variable

# Configuration
GEMINI_API_KEY = "AIzaSyDb_dgJI1gxqYGD6xEW5wEiCTEJjyy6z3U"
MODEL = "gemini-2.0-flash"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"
WORKSPACE_DIR = os.path.join(os.getcwd(), "workspace")
UPLOAD_DIR = os.path.join(os.getcwd(), "uploads")
AGENT_SESSIONS = {}

# Ensure directories exist
os.makedirs(WORKSPACE_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Mount static files for uploaded content
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

# Pydantic Models
class UserSignup(BaseModel):
    name: str
    email: str
    password: str

class UserLogin(BaseModel):
    email: str
    password: str

class UserResponse(BaseModel):
    id: str
    name: str
    email: str
    created_at: datetime

class ChatRequest(BaseModel):
    message: str
    system_prompt: Optional[str] = None
    conversation_history: Optional[List[Dict]] = None
    session_id: Optional[str] = None

class FileOperation(BaseModel):
    action: str  # create, read, update, delete, list
    path: Optional[str] = None
    content: Optional[str] = None
    session_id: Optional[str] = None

class TerminalRequest(BaseModel):
    command: str
    working_dir: Optional[str] = WORKSPACE_DIR
    session_id: Optional[str] = None

class AgentTask(BaseModel):
    task: str
    session_id: Optional[str] = None
    auto_execute: bool = True

class AgentPlan(BaseModel):
    steps: List[Dict]
    session_id: str

class GitHubCloneRequest(BaseModel):
    repo_url: str
    target_dir: Optional[str] = None
    session_id: Optional[str] = None

class UploadResponse(BaseModel):
    filename: str
    path: str
    size: int
    file_count: Optional[int] = None

class ProjectRequirements(BaseModel):
    requirements: str
    session_id: str
    project_path: str
    auto_execute: bool = True

class CSSUpdateRequest(BaseModel):
    session_id: str
    project_path: str
    css_requirements: str
    target_files: Optional[List[str]] = None

class BugFixRequest(BaseModel):
    error_description: str
    code_snippet: str
    session_id: str
    project_path: str

class ChatWithProjectRequest(BaseModel):
    message: str
    session_id: str
    project_path: Optional[str] = None
    auto_execute: bool = True

class FileDiffRequest(BaseModel):
    file_path: str
    old_content: str
    new_content: str
    session_id: str
    user_id: str

class FileDiffResponse(BaseModel):
    diff_id: str
    file_path: str
    diff_content: str
    changes: List[str]
    created_at: datetime

# Authentication Utilities
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return hash_password(plain_password) == hashed_password

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(hours=24)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm="HS256")
    return encoded_jwt

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=["HS256"])
        user_id = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")
        user = await users_collection.find_one({"_id": ObjectId(user_id)})
        if user is None:
            raise HTTPException(status_code=401, detail="User not found")
        return user
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

# WebSocket Connection Manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.terminal_processes: Dict[str, Any] = {}
        self.chat_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections.append(websocket)
        self.chat_connections[client_id] = websocket

    def disconnect(self, websocket: WebSocket, client_id: str):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        if client_id in self.chat_connections:
            del self.chat_connections[client_id]
        if client_id in self.terminal_processes:
            del self.terminal_processes[client_id]

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

    async def start_terminal_process(self, client_id: str, working_dir: str = WORKSPACE_DIR):
        """Start a terminal process for a client"""
        try:
            process = await asyncio.create_subprocess_shell(
                '/bin/bash' if os.name != 'nt' else 'cmd.exe',
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=working_dir
            )
            self.terminal_processes[client_id] = process
            return process
        except Exception as e:
            print(f"Error starting terminal process: {e}")
            return None

    async def execute_terminal_command(self, client_id: str, command: str):
        """Execute a command in the terminal process"""
        if client_id not in self.terminal_processes:
            await self.start_terminal_process(client_id)
        
        process = self.terminal_processes[client_id]
        try:
            process.stdin.write(f"{command}\n".encode())
            await process.stdin.drain()
            
            # Read output with timeout
            output = ""
            while True:
                try:
                    line = await asyncio.wait_for(process.stdout.readline(), timeout=1.0)
                    if not line:
                        break
                    output += line.decode()
                except asyncio.TimeoutError:
                    break
            
            return output
        except Exception as e:
            return f"Error executing command: {str(e)}"

manager = ConnectionManager()

# File Diff Manager
class FileDiffManager:
    @staticmethod
    async def create_diff(request: FileDiffRequest) -> FileDiffResponse:
        """Create a unified diff between old and new content"""
        old_lines = request.old_content.splitlines(keepends=True)
        new_lines = request.new_content.splitlines(keepends=True)
        
        diff_content = ''.join(unified_diff(
            old_lines, 
            new_lines,
            fromfile=f'a/{request.file_path}',
            tofile=f'b/{request.file_path}',
            lineterm='\n'
        ))
        
        # Extract individual changes
        changes = []
        for line in diff_content.split('\n'):
            if line.startswith('+') and not line.startswith('+++'):
                changes.append(f"Added: {line[1:]}")
            elif line.startswith('-') and not line.startswith('---'):
                changes.append(f"Removed: {line[1:]}")
            elif line.startswith('@@'):
                changes.append(f"Context: {line}")
        
        diff_data = {
            "file_path": request.file_path,
            "old_content": request.old_content,
            "new_content": request.new_content,
            "diff_content": diff_content,
            "changes": changes,
            "session_id": request.session_id,
            "user_id": request.user_id,
            "created_at": datetime.utcnow()
        }
        
        result = await file_diffs_collection.insert_one(diff_data)
        diff_data["diff_id"] = str(result.inserted_id)
        
        return FileDiffResponse(**diff_data)
    
    @staticmethod
    async def get_diff(diff_id: str):
        """Get a specific diff by ID"""
        diff = await file_diffs_collection.find_one({"_id": ObjectId(diff_id)})
        if diff:
            diff["diff_id"] = str(diff["_id"])
            return FileDiffResponse(**diff)
        return None
    
    @staticmethod
    async def get_diffs_by_session(session_id: str, limit: int = 50):
        """Get diffs for a session"""
        cursor = file_diffs_collection.find({"session_id": session_id}).sort("created_at", -1).limit(limit)
        diffs = []
        async for diff in cursor:
            diff["diff_id"] = str(diff["_id"])
            diffs.append(FileDiffResponse(**diff))
        return diffs

# Gemini API Helper
async def call_gemini_with_timeout(
    model: str, 
    payload: Dict[str, Any], 
    timeout_ms: int = 30000
) -> Dict[str, Any]:
    url = f"{GEMINI_BASE_URL}/{model}:generateContent"
    
    body = {
        "system_instruction": {"parts": [{"text": payload.get("systemText", "")}]} 
        if payload.get("systemText") else None,
        "contents": payload.get("contents", [])
    }
    
    body = {k: v for k, v in body.items() if v is not None}
    
    headers = {
        "Content-Type": "application/json",
        "X-goog-api-key": GEMINI_API_KEY
    }
    
    timeout = httpx.Timeout(timeout_ms / 1000)
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, json=body, timeout=timeout)
            
            if response.status_code != 200:
                error_detail = f"Gemini API error: {response.status_code} - {response.text}"
                raise HTTPException(status_code=response.status_code, detail=error_detail)
            
            return response.json()
            
    except httpx.TimeoutException:
        raise HTTPException(status_code=408, detail="Request timeout")
    except httpx.RequestError as e:
        raise HTTPException(status_code=500, detail=f"Request error: {str(e)}")

# GitHub Repository Manager
class GitHubManager:
    @staticmethod
    async def clone_repository(repo_url: str, target_dir: str = None, session_id: str = None) -> Dict:
        """Clone a GitHub repository to workspace"""
        try:
            if not target_dir:
                # Extract repo name from URL
                repo_name = repo_url.split('/')[-1].replace('.git', '')
                target_dir = os.path.join(WORKSPACE_DIR, f"repo_{session_id or uuid.uuid4()}_{repo_name}")
            else:
                target_dir = os.path.join(WORKSPACE_DIR, target_dir)
            
            # Remove if already exists
            if os.path.exists(target_dir):
                shutil.rmtree(target_dir)
            
            # Clone repository
            print(f"Cloning {repo_url} to {target_dir}")
            Repo.clone_from(repo_url, target_dir)
            
            # Get repository info
            repo = Repo(target_dir)
            branch = repo.active_branch.name
            commit = repo.head.commit.hexsha[:7]
            
            # Count files
            file_count = 0
            for root, dirs, files in os.walk(target_dir):
                file_count += len(files)
            
            return {
                "success": True,
                "repo_url": repo_url,
                "local_path": target_dir,
                "branch": branch,
                "latest_commit": commit,
                "file_count": file_count,
                "message": f"Successfully cloned repository to {target_dir}"
            }
            
        except git.exc.GitCommandError as e:
            raise HTTPException(status_code=400, detail=f"Git clone failed: {str(e)}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Repository cloning error: {str(e)}")
    
    @staticmethod
    async def get_repo_info(repo_path: str) -> Dict:
        """Get information about a cloned repository"""
        try:
            repo = Repo(repo_path)
            return {
                "branch": repo.active_branch.name,
                "latest_commit": repo.head.commit.hexsha[:7],
                "commit_message": repo.head.commit.message,
                "author": str(repo.head.commit.author),
                "committed_date": repo.head.commit.committed_date,
                "is_dirty": repo.is_dirty(),
                "untracked_files": repo.untracked_files
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to get repo info: {str(e)}")

# File Upload and Extraction Manager
class UploadManager:
    @staticmethod
    async def save_uploaded_file(file: UploadFile, session_id: str = None) -> Dict:
        """Save uploaded file and return its info"""
        try:
            # Create session directory
            session_upload_dir = os.path.join(UPLOAD_DIR, f"session_{session_id or uuid.uuid4()}")
            os.makedirs(session_upload_dir, exist_ok=True)
            
            # Save file
            file_path = os.path.join(session_upload_dir, file.filename)
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            return {
                "filename": file.filename,
                "path": file_path,
                "size": len(content),
                "session_dir": session_upload_dir
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")
    
    @staticmethod
    async def extract_archive(file_path: str, extract_to: str = None) -> Dict:
        """Extract zip/tar archive and return file info"""
        try:
            if not extract_to:
                extract_to = file_path + "_extracted"
            
            if os.path.exists(extract_to):
                shutil.rmtree(extract_to)
            os.makedirs(extract_to, exist_ok=True)
            
            file_count = 0
            extracted_files = []
            
            if file_path.endswith('.zip'):
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_to)
                    file_count = len(zip_ref.namelist())
                    extracted_files = zip_ref.namelist()
                    
            elif file_path.endswith(('.tar', '.tar.gz', '.tgz')):
                with tarfile.open(file_path, 'r:*') as tar_ref:
                    tar_ref.extractall(extract_to)
                    file_count = len(tar_ref.getnames())
                    extracted_files = tar_ref.getnames()
            
            else:
                raise HTTPException(status_code=400, detail="Unsupported archive format")
            
            return {
                "extracted_to": extract_to,
                "file_count": file_count,
                "extracted_files": extracted_files[:10]  # First 10 files
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Archive extraction failed: {str(e)}")
    
    @staticmethod
    async def copy_to_workspace(source_dir: str, target_dir: str = None) -> Dict:
        """Copy uploaded/extracted files to workspace"""
        try:
            if not target_dir:
                target_dir = os.path.join(WORKSPACE_DIR, f"project_{uuid.uuid4()}")
            
            if os.path.exists(target_dir):
                shutil.rmtree(target_dir)
            
            shutil.copytree(source_dir, target_dir)
            
            # Count files
            file_count = 0
            for root, dirs, files in os.walk(target_dir):
                file_count += len(files)
            
            return {
                "workspace_path": target_dir,
                "file_count": file_count,
                "message": f"Successfully copied {file_count} files to workspace"
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Copy to workspace failed: {str(e)}")

# File System Operations (Enhanced with Diff Support)
class FileManager:
    def __init__(self, base_dir=WORKSPACE_DIR):
        self.base_dir = base_dir
    
    def get_absolute_path(self, path: str) -> str:
        """Convert relative path to absolute path within workspace"""
        if not path:
            return self.base_dir
        absolute_path = os.path.join(self.base_dir, path.lstrip('/'))
        # Security check to prevent directory traversal
        if not absolute_path.startswith(self.base_dir):
            raise HTTPException(status_code=403, detail="Access denied")
        return absolute_path
    
    async def create_file(self, path: str, content: str = "") -> Dict:
        abs_path = self.get_absolute_path(path)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        
        with open(abs_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return {"status": "created", "path": path, "size": len(content)}
    
    async def read_file(self, path: str) -> Dict:
        abs_path = self.get_absolute_path(path)
        
        if not os.path.exists(abs_path):
            raise HTTPException(status_code=404, detail="File not found")
        
        with open(abs_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return {"path": path, "content": content, "size": len(content)}
    
    async def update_file_with_diff(self, path: str, new_content: str, session_id: str, user_id: str) -> Dict:
        """Update file and create diff preview"""
        abs_path = self.get_absolute_path(path)
        
        if not os.path.exists(abs_path):
            raise HTTPException(status_code=404, detail="File not found")
        
        # Read current content
        with open(abs_path, 'r', encoding='utf-8') as f:
            old_content = f.read()
        
        # Create diff preview
        diff_request = FileDiffRequest(
            file_path=path,
            old_content=old_content,
            new_content=new_content,
            session_id=session_id,
            user_id=user_id
        )
        
        diff_response = await FileDiffManager.create_diff(diff_request)
        
        # Return diff instead of immediately updating
        return {
            "status": "diff_created",
            "path": path,
            "diff_id": diff_response.diff_id,
            "diff_content": diff_response.diff_content,
            "changes": diff_response.changes,
            "message": "File diff created. Use /api/files/apply-diff to apply changes."
        }
    
    async def apply_diff(self, diff_id: str) -> Dict:
        """Apply a diff to update the file"""
        diff = await FileDiffManager.get_diff(diff_id)
        if not diff:
            raise HTTPException(status_code=404, detail="Diff not found")
        
        abs_path = self.get_absolute_path(diff.file_path)
        
        # Update the file with new content
        with open(abs_path, 'w', encoding='utf-8') as f:
            f.write(diff.new_content)
        
        return {
            "status": "updated",
            "path": diff.file_path,
            "diff_id": diff_id,
            "size": len(diff.new_content),
            "message": "File updated successfully"
        }
    
    async def delete_file(self, path: str) -> Dict:
        abs_path = self.get_absolute_path(path)
        
        if not os.path.exists(abs_path):
            raise HTTPException(status_code=404, detail="File not found")
        
        if os.path.isdir(abs_path):
            shutil.rmtree(abs_path)
            action = "directory deleted"
        else:
            os.remove(abs_path)
            action = "file deleted"
        
        return {"status": action, "path": path}
    
    async def list_files(self, path: str = "") -> Dict:
        abs_path = self.get_absolute_path(path)
        
        if not os.path.exists(abs_path):
            raise HTTPException(status_code=404, detail="Path not found")
        
        items = []
        for item in os.listdir(abs_path):
            item_path = os.path.join(abs_path, item)
            items.append({
                "name": item,
                "path": os.path.join(path, item) if path else item,
                "type": "directory" if os.path.isdir(item_path) else "file",
                "size": os.path.getsize(item_path) if os.path.isfile(item_path) else 0,
                "modified": os.path.getmtime(item_path)
            })
        
        return {"path": path, "items": items}

# Terminal Operations
class TerminalManager:
    @staticmethod
    async def execute_command(command: str, working_dir: str = WORKSPACE_DIR) -> Dict:
        try:
            # Change to working directory
            original_dir = os.getcwd()
            os.chdir(working_dir)
            
            # Execute command
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            stdout, stderr = process.communicate()
            
            # Return to original directory
            os.chdir(original_dir)
            
            return {
                "command": command,
                "stdout": stdout,
                "stderr": stderr,
                "return_code": process.returncode,
                "success": process.returncode == 0
            }
            
        except Exception as e:
            os.chdir(original_dir)
            raise HTTPException(status_code=500, detail=f"Command execution failed: {str(e)}")

# Enhanced Autonomous Agent System
class AdvancedAutonomousAgent:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.file_manager = FileManager()
        self.terminal = TerminalManager()
        self.conversation_history = []
        self.project_context = {}
        self.current_project_path = None
        
        # Initialize agent workspace
        self.agent_workspace = os.path.join(WORKSPACE_DIR, f"agent_{session_id}")
        os.makedirs(self.agent_workspace, exist_ok=True)
    
    async def understand_project_structure(self, project_path: str) -> Dict:
        """Deep analysis of project structure and dependencies"""
        try:
            # Read package.json for React projects
            package_json_path = os.path.join(project_path, 'package.json')
            dependencies = {}
            if os.path.exists(package_json_path):
                with open(package_json_path, 'r') as f:
                    package_data = json.load(f)
                    dependencies = {
                        **package_data.get('dependencies', {}),
                        **package_data.get('devDependencies', {})
                    }
            
            # Analyze project structure
            project_files = []
            tech_stack = []
            
            for root, dirs, files in os.walk(project_path):
                for file in files:
                    rel_path = os.path.relpath(os.path.join(root, file), project_path)
                    project_files.append(rel_path)
                    
                    # Detect technology stack
                    if file == 'package.json':
                        tech_stack.append('nodejs')
                    elif file.endswith('.jsx') or file.endswith('.tsx'):
                        tech_stack.append('react')
                    elif file.endswith('.vue'):
                        tech_stack.append('vue')
                    elif file.endswith('.py'):
                        tech_stack.append('python')
                    elif file.endswith('.css') or file.endswith('.scss'):
                        tech_stack.append('css')
                    elif file.endswith('.html'):
                        tech_stack.append('html')
            
            self.project_context = {
                'project_path': project_path,
                'dependencies': dependencies,
                'tech_stack': list(set(tech_stack)),
                'file_count': len(project_files),
                'structure': project_files[:50]  # First 50 files
            }
            
            self.current_project_path = project_path
            return self.project_context
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Project analysis failed: {str(e)}")
    
    async def generate_implementation_plan(self, requirements: str, project_context: Dict) -> Dict:
        """Generate detailed implementation plan based on requirements"""
        system_prompt = """You are an expert full-stack developer. Analyze the requirements and create a detailed implementation plan considering the existing project structure.

        Available Context:
        - Project Path: {project_path}
        - Tech Stack: {tech_stack}
        - Dependencies: {dependencies}
        - File Count: {file_count}

        Requirements: {requirements}

        Create a step-by-step implementation plan including:
        1. File modifications needed
        2. New files to create
        3. Dependencies to install
        4. Code changes required
        5. CSS/styling updates
        6. Testing steps

        Return as JSON with this structure:
        {{
            "analysis": "analysis of requirements",
            "implementation_steps": [
                {{
                    "step": 1,
                    "type": "file_operation|terminal|code_change|css_update",
                    "action": "create_file|update_file|install_deps|run_command",
                    "file_path": "path/to/file",
                    "content": "code/content if applicable",
                    "description": "detailed description",
                    "command": "terminal command if applicable"
                }}
            ],
            "new_dependencies": ["package1", "package2"],
            "estimated_time": "estimate",
            "risk_level": "low|medium|high"
        }}""".format(
            project_path=project_context.get('project_path', ''),
            tech_stack=project_context.get('tech_stack', []),
            dependencies=project_context.get('dependencies', {}),
            file_count=project_context.get('file_count', 0),
            requirements=requirements
        )
        
        contents = [{
            "role": "user",
            "parts": [{"text": f"Create implementation plan for: {requirements}"}]
        }]
        
        payload = {
            "systemText": system_prompt,
            "contents": contents
        }
        
        response = await call_gemini_with_timeout(MODEL, payload)
        
        if "candidates" in response and len(response["candidates"]) > 0:
            candidate = response["candidates"][0]
            if "content" in candidate and "parts" in candidate["content"]:
                plan_text = candidate["content"]["parts"][0].get("text", "")
                
                try:
                    # Extract JSON from response
                    start_idx = plan_text.find('{')
                    end_idx = plan_text.rfind('}') + 1
                    if start_idx != -1 and end_idx != -1:
                        json_str = plan_text[start_idx:end_idx]
                        plan_data = json.loads(json_str)
                        return plan_data
                except json.JSONDecodeError:
                    # Fallback for non-JSON response
                    return {
                        "analysis": plan_text,
                        "implementation_steps": [
                            {
                                "step": 1,
                                "type": "analysis",
                                "action": "Analyze requirements",
                                "description": plan_text
                            }
                        ],
                        "new_dependencies": [],
                        "estimated_time": "Unknown",
                        "risk_level": "medium"
                    }
        
        raise HTTPException(status_code=500, detail="Failed to generate implementation plan")
    
    async def execute_implementation_plan(self, plan: Dict, user_id: str) -> Dict:
        """Execute the implementation plan step by step with diff previews"""
        results = []
        project_path = self.project_context.get('project_path', WORKSPACE_DIR)
        
        for step in plan.get('implementation_steps', []):
            step_result = {
                "step": step.get('step'),
                "type": step.get('type'),
                "action": step.get('action'),
                "file_path": step.get('file_path'),
                "status": "pending",
                "result": None,
                "error": None
            }
            
            try:
                if step.get('type') == 'file_operation':
                    if step.get('action') == 'create_file':
                        # CREATE NEW FILE
                        file_path = step.get('file_path', '')
                        content = step.get('content', '')
                        result = await self.file_manager.create_file(file_path, content)
                        step_result["status"] = "completed"
                        step_result["result"] = f"Created file: {file_path} ({len(content)} bytes)"
                        
                    elif step.get('action') == 'update_file':
                        # UPDATE EXISTING FILE WITH DIFF PREVIEW
                        file_path = step.get('file_path', '')
                        content = step.get('content', '')
                        result = await self.file_manager.update_file_with_diff(
                            file_path, content, self.session_id, user_id
                        )
                        step_result["status"] = "diff_created"
                        step_result["result"] = result
                
                elif step.get('type') == 'terminal':
                    # Execute terminal commands
                    command = step.get('command', '')
                    if command:
                        command_result = await self.terminal.execute_command(command, project_path)
                        step_result["result"] = command_result
                        step_result["status"] = "completed" if command_result.get('success') else "failed"
                
                elif step.get('type') == 'code_change':
                    # Handle code modifications in existing files with diff preview
                    file_path = step.get('file_path', '')
                    content = step.get('content', '')
                    if file_path and content:
                        # Check if file exists to decide between create or update
                        abs_path = os.path.join(project_path, file_path)
                        if os.path.exists(abs_path):
                            result = await self.file_manager.update_file_with_diff(
                                file_path, content, self.session_id, user_id
                            )
                            step_result["action"] = "updated_file_with_diff"
                        else:
                            result = await self.file_manager.create_file(file_path, content)
                            step_result["action"] = "created_file"
                        
                        step_result["status"] = "completed" if step_result["action"] == "created_file" else "diff_created"
                        step_result["result"] = result
                
                elif step.get('type') == 'css_update':
                    # Special handling for CSS updates with diff preview
                    css_result = await self.update_css_styles(
                        step.get('file_path', ''),
                        step.get('content', ''),
                        step.get('description', ''),
                        user_id
                    )
                    step_result["result"] = css_result
                    step_result["status"] = "diff_created" if "diff_id" in css_result else "completed"
                
                results.append(step_result)
                
            except Exception as e:
                step_result["status"] = "failed"
                step_result["error"] = str(e)
                results.append(step_result)
        
        return {
            "success": True,
            "steps_executed": len(results),
            "steps_completed": len([r for r in results if r["status"] == "completed"]),
            "steps_with_diff": len([r for r in results if r["status"] == "diff_created"]),
            "steps_failed": len([r for r in results if r["status"] == "failed"]),
            "detailed_results": results
        }
    
    async def update_css_styles(self, file_path: str, new_css: str, description: str, user_id: str) -> Dict:
        """Update CSS files with professional styles - WITH DIFF PREVIEW"""
        try:
            project_path = self.project_context.get('project_path', WORKSPACE_DIR)
            abs_path = os.path.join(project_path, file_path)
            
            if os.path.exists(abs_path):
                # UPDATE EXISTING FILE - Read current content and create diff
                with open(abs_path, 'r', encoding='utf-8') as f:
                    existing_css = f.read()
                
                # For now, append new CSS. In production, you'd want smarter merging
                updated_css = existing_css + "\n\n/* Professional CSS Update */\n" + new_css
                
                # Create diff preview instead of directly updating
                diff_request = FileDiffRequest(
                    file_path=file_path,
                    old_content=existing_css,
                    new_content=updated_css,
                    session_id=self.session_id,
                    user_id=user_id
                )
                
                diff_response = await FileDiffManager.create_diff(diff_request)
                
                return {
                    "action": "css_update_diff",
                    "file": file_path,
                    "diff_id": diff_response.diff_id,
                    "changes_made": description,
                    "existing_content_modified": True,
                    "new_content_added": len(new_css),
                    "diff_content": diff_response.diff_content
                }
            else:
                # CREATE NEW FILE - File doesn't exist, so create it
                # Ensure directory exists
                os.makedirs(os.path.dirname(abs_path), exist_ok=True)
                
                with open(abs_path, 'w', encoding='utf-8') as f:
                    f.write(new_css)
                
                return {
                    "action": "css_created",
                    "file": file_path,
                    "changes_made": description,
                    "existing_content_modified": False,
                    "new_content_added": len(new_css)
                }
                
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"CSS update failed: {str(e)}")
    
    async def implement_ecommerce_features(self, requirements: str, user_id: str) -> Dict:
        """Specialized method for e-commerce implementations"""
        system_prompt = """You are an expert e-commerce developer. Implement dynamic e-commerce features including:
        - Product listings and filtering
        - Shopping cart functionality
        - User authentication
        - Payment integration setup
        - Order management
        - Responsive product pages
        
        Provide complete React components, state management, and professional CSS."""
        
        plan = await self.generate_implementation_plan(requirements, self.project_context)
        execution_results = await self.execute_implementation_plan(plan, user_id)
        
        return {
            "ecommerce_plan": plan,
            "execution_results": execution_results,
            "features_implemented": ["product_catalog", "shopping_cart", "responsive_design"]
        }

# Enhanced Chat Handler with Code Execution
class ChatHandler:
    def __init__(self):
        self.file_manager = FileManager()
        self.terminal = TerminalManager()
    
    async def process_chat_request(self, request: ChatRequest, project_path: str = None, user_id: str = None) -> Dict:
        """Process chat messages and execute code changes when needed"""
        
        # Analyze if the message contains actionable requests
        action_analysis = await self.analyze_user_intent(request.message, project_path)
        
        if action_analysis.get('requires_execution', False) and request.session_id:
            # Execute the requested changes
            execution_result = await self.execute_chat_actions(action_analysis, request.session_id, project_path, user_id)
            
            # Generate response including execution results
            response_text = await self.generate_response_with_results(request.message, execution_result)
            
            return {
                "success": True,
                "response": response_text,
                "session_id": request.session_id,
                "execution_performed": True,
                "execution_results": execution_result,
                "actions_taken": action_analysis.get('actions', [])
            }
        else:
            # Regular chat response
            response = await self.generate_chat_response(request)
            return {
                "success": True,
                "response": response,
                "session_id": request.session_id,
                "execution_performed": False
            }
    
    async def analyze_user_intent(self, message: str, project_path: str = None) -> Dict:
        """Analyze if user wants to execute code changes, fix bugs, update CSS, etc."""
        
        system_prompt = """Analyze the user's message and determine if they are requesting specific code changes, bug fixes, CSS updates, or functionality implementations.

        Common request patterns:
        - "Fix the button not working" → bug fix
        - "Change the background color to blue" → CSS update    
        - "Add a login form" → new functionality
        - "The cart is not updating properly" → bug fix
        - "Make the header sticky" → CSS update
        - "Create a product details page" → new component
        - "Update the API endpoint" → code change
        - "The form validation is broken" → bug fix
        - "Improve the mobile responsiveness" → CSS update

        Return JSON analysis:
        {
            "requires_execution": true/false,
            "intent": "css_update|bug_fix|new_feature|code_change|general_chat",
            "actions": [
                {
                    "type": "update_css|create_file|update_file|fix_bug|run_command",
                    "target": "file_path or component",
                    "description": "what needs to be done",
                    "priority": "high|medium|low"
                }
            ],
            "files_affected": ["file1.css", "file2.jsx"],
            "urgency": "immediate|soon|later"
        }"""
        
        contents = [{"role": "user", "parts": [{"text": f"User message: {message}"}]}]
        payload = {"systemText": system_prompt, "contents": contents}
        
        response = await call_gemini_with_timeout(MODEL, payload)
        
        if "candidates" in response and len(response["candidates"]) > 0:
            candidate = response["candidates"][0]
            if "content" in candidate and "parts" in candidate["content"]:
                analysis_text = candidate["content"]["parts"][0].get("text", "")
                
                try:
                    start_idx = analysis_text.find('{')
                    end_idx = analysis_text.rfind('}') + 1
                    if start_idx != -1 and end_idx != -1:
                        return json.loads(analysis_text[start_idx:end_idx])
                except:
                    pass
        
        # Default analysis for general chat
        return {"requires_execution": False, "intent": "general_chat"}
    
    async def execute_chat_actions(self, analysis: Dict, session_id: str, project_path: str, user_id: str) -> Dict:
        """Execute the actions identified from user chat"""
        results = []
        
        if session_id in AGENT_SESSIONS:
            agent = AGENT_SESSIONS[session_id]
            
            for action in analysis.get('actions', []):
                try:
                    if action['type'] == 'update_css':
                        result = await self.handle_css_update(action, agent, project_path, user_id)
                        results.append(result)
                    
                    elif action['type'] == 'fix_bug':
                        result = await self.handle_bug_fix(action, agent, project_path, user_id)
                        results.append(result)
                    
                    elif action['type'] == 'create_file':
                        result = await self.handle_create_file(action, agent)
                        results.append(result)
                    
                    elif action['type'] == 'update_file':
                        result = await self.handle_update_file(action, agent, user_id)
                        results.append(result)
                    
                    elif action['type'] == 'run_command':
                        result = await self.handle_terminal_command(action, project_path)
                        results.append(result)
                
                except Exception as e:
                    results.append({
                        "action": action.get('type'),
                        "status": "failed",
                        "error": str(e)
                    })
        
        return {"actions_executed": results}
    
    async def handle_css_update(self, action: Dict, agent: AdvancedAutonomousAgent, project_path: str, user_id: str) -> Dict:
        """Handle CSS update requests from chat"""
        css_requirements = action.get('description', '')
        target_file = action.get('target', 'styles.css')
        
        # Generate professional CSS
        system_prompt = f"""You are a CSS expert. Create professional CSS code for: {css_requirements}
        
        Requirements: {css_requirements}
        Target: {target_file}
        
        Generate clean, modern CSS that follows best practices."""
        
        contents = [{"role": "user", "parts": [{"text": f"Create CSS for: {css_requirements}"}]}]
        payload = {"systemText": system_prompt, "contents": contents}
        
        response = await call_gemini_with_timeout(MODEL, payload)
        
        if "candidates" in response and len(response["candidates"]) > 0:
            candidate = response["candidates"][0]
            if "content" in candidate and "parts" in candidate["content"]:
                css_code = candidate["content"]["parts"][0].get("text", "")
                
                # Execute the CSS update with diff preview
                result = await agent.update_css_styles(target_file, css_code, css_requirements, user_id)
                return {
                    "action": "css_update",
                    "target": target_file,
                    "status": "completed" if "diff_id" not in result else "diff_created",
                    "result": result
                }
        
        return {"action": "css_update", "status": "failed", "error": "Failed to generate CSS"}
    
    async def handle_bug_fix(self, action: Dict, agent: AdvancedAutonomousAgent, project_path: str, user_id: str) -> Dict:
        """Handle bug fix requests from chat"""
        bug_description = action.get('description', '')
        target_file = action.get('target', '')
        
        system_prompt = f"""You are a debugging expert. Analyze and fix this bug: {bug_description}
        
        Provide the complete fixed code for the affected file.
        Return JSON with:
        {{
            "analysis": "bug analysis",
            "fixed_code": "complete fixed code",
            "file_path": "path/to/file",
            "changes_made": "description of fixes"
        }}"""
        
        contents = [{"role": "user", "parts": [{"text": f"Fix this bug: {bug_description}"}]}]
        payload = {"systemText": system_prompt, "contents": contents}
        
        response = await call_gemini_with_timeout(MODEL, payload)
        
        if "candidates" in response and len(response["candidates"]) > 0:
            candidate = response["candidates"][0]
            if "content" in candidate and "parts" in candidate["content"]:
                fix_text = candidate["content"]["parts"][0].get("text", "")
                
                try:
                    start_idx = fix_text.find('{')
                    end_idx = fix_text.rfind('}') + 1
                    if start_idx != -1 and end_idx != -1:
                        fix_data = json.loads(fix_text[start_idx:end_idx])
                        
                        # Update the file with fixed code using diff preview
                        result = await agent.file_manager.update_file_with_diff(
                            fix_data.get('file_path', target_file),
                            fix_data.get('fixed_code', ''),
                            agent.session_id,
                            user_id
                        )
                        
                        return {
                            "action": "bug_fix",
                            "target": fix_data.get('file_path', target_file),
                            "status": "diff_created",
                            "analysis": fix_data.get('analysis'),
                            "changes": fix_data.get('changes_made'),
                            "result": result
                        }
                except:
                    # Fallback: return the text response
                    return {
                        "action": "bug_fix",
                        "status": "analysis_only",
                        "analysis": fix_text
                    }
        
        return {"action": "bug_fix", "status": "failed"}
    
    async def handle_create_file(self, action: Dict, agent: AdvancedAutonomousAgent) -> Dict:
        """Handle file creation requests"""
        file_path = action.get('target', '')
        description = action.get('description', '')
        
        system_prompt = f"""Create a new file with this functionality: {description}
        
        File path: {file_path}
        Return the complete file content."""
        
        contents = [{"role": "user", "parts": [{"text": f"Create file content for: {description}"}]}]
        payload = {"systemText": system_prompt, "contents": contents}
        
        response = await call_gemini_with_timeout(MODEL, payload)
        
        if "candidates" in response and len(response["candidates"]) > 0:
            candidate = response["candidates"][0]
            if "content" in candidate and "parts" in candidate["content"]:
                file_content = candidate["content"]["parts"][0].get("text", "")
                
                result = await agent.file_manager.create_file(file_path, file_content)
                return {
                    "action": "create_file",
                    "target": file_path,
                    "status": "completed",
                    "result": result
                }
        
        return {"action": "create_file", "status": "failed"}
    
    async def handle_update_file(self, action: Dict, agent: AdvancedAutonomousAgent, user_id: str) -> Dict:
        """Handle file update requests"""
        file_path = action.get('target', '')
        description = action.get('description', '')
        
        # First read the current file content
        try:
            current_content = await agent.file_manager.read_file(file_path)
            current_code = current_content.get('content', '')
        except:
            current_code = ""
        
        system_prompt = f"""Update this file with: {description}
        
        Current file content:
        {current_code}
        
        Return the complete updated file content."""
        
        contents = [{"role": "user", "parts": [{"text": f"Update the file with: {description}"}]}]
        payload = {"systemText": system_prompt, "contents": contents}
        
        response = await call_gemini_with_timeout(MODEL, payload)
        
        if "candidates" in response and len(response["candidates"]) > 0:
            candidate = response["candidates"][0]
            if "content" in candidate and "parts" in candidate["content"]:
                updated_content = candidate["content"]["parts"][0].get("text", "")
                
                result = await agent.file_manager.update_file_with_diff(file_path, updated_content, agent.session_id, user_id)
                return {
                    "action": "update_file",
                    "target": file_path,
                    "status": "diff_created",
                    "result": result
                }
        
        return {"action": "update_file", "status": "failed"}
    
    async def handle_terminal_command(self, action: Dict, project_path: str) -> Dict:
        """Handle terminal command execution"""
        command = action.get('description', '')
        result = await self.terminal.execute_command(command, project_path)
        return {
            "action": "run_command",
            "command": command,
            "status": "completed" if result.get('success') else "failed",
            "result": result
        }
    
    async def generate_response_with_results(self, user_message: str, execution_results: Dict) -> str:
        """Generate a natural language response including execution results"""
        
        system_prompt = """You are a helpful AI assistant. Summarize what you've done based on the execution results and provide a friendly response to the user.

        Include:
        1. What changes were made
        2. Any files created or updated
        3. Next steps if needed
        4. Keep it conversational and helpful"""
        
        results_summary = json.dumps(execution_results, indent=2)
        
        contents = [{
            "role": "user", 
            "parts": [{
                "text": f"User asked: {user_message}\n\nExecution Results:\n{results_summary}\n\nProvide a helpful summary response:"
            }]
        }]
        
        payload = {"systemText": system_prompt, "contents": contents}
        
        response = await call_gemini_with_timeout(MODEL, payload)
        
        if "candidates" in response and len(response["candidates"]) > 0:
            candidate = response["candidates"][0]
            if "content" in candidate and "parts" in candidate["content"]:
                return candidate["content"]["parts"][0].get("text", "")
        
        return "I've processed your request and made the necessary changes to your project."
    
    async def generate_chat_response(self, request: ChatRequest) -> str:
        """Generate regular chat response"""
        contents = []
        
        # Add conversation history
        if request.conversation_history:
            for msg in request.conversation_history:
                if isinstance(msg, dict):
                    role = "user" if msg.get("role") in ["user", "human"] else "model"
                    content = msg.get("content", "")
                    contents.append({"role": role, "parts": [{"text": content}]})
        
        # Add current message
        contents.append({"role": "user", "parts": [{"text": request.message}]})
        
        payload = {
            "systemText": request.system_prompt,
            "contents": contents
        }
        
        response = await call_gemini_with_timeout(MODEL, payload)
        
        if "candidates" in response and len(response["candidates"]) > 0:
            candidate = response["candidates"][0]
            if "content" in candidate and "parts" in candidate["content"]:
                return candidate["content"]["parts"][0].get("text", "")
        
        return "I'm here to help! How can I assist you with your project?"

# API Routes

@app.get("/")
async def root():
    return {
        "message": "Cursor AI Clone API v4.0", 
        "status": "running", 
        "features": [
            "github", "upload", "agents", "terminal", "chat", 
            "auto_code_updates", "chat_execution", "bug_fixes",
            "websocket_terminal", "websocket_chat", "file_diff_preview",
            "user_authentication", "session_persistence"
        ]
    }

# Authentication Endpoints
@app.post("/api/auth/signup")
async def signup(user: UserSignup):
    """User registration endpoint"""
    # Check if user already exists
    existing_user = await users_collection.find_one({"email": user.email})
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Create new user
    user_data = {
        "name": user.name,
        "email": user.email,
        "password": hash_password(user.password),
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }
    
    result = await users_collection.insert_one(user_data)
    user_data["id"] = str(result.inserted_id)
    
    # Create access token
    access_token = create_access_token(data={"sub": str(result.inserted_id)})
    
    return {
        "success": True,
        "message": "User created successfully",
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "id": user_data["id"],
            "name": user.name,
            "email": user.email
        }
    }

@app.post("/api/auth/login")
async def login(user: UserLogin):
    """User login endpoint"""
    # Find user by email
    db_user = await users_collection.find_one({"email": user.email})
    if not db_user or not verify_password(user.password, db_user["password"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    
    # Create access token
    access_token = create_access_token(data={"sub": str(db_user["_id"])})
    
    return {
        "success": True,
        "message": "Login successful",
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "id": str(db_user["_id"]),
            "name": db_user["name"],
            "email": db_user["email"]
        }
    }

@app.get("/api/auth/me")
async def get_current_user_profile(current_user: dict = Depends(get_current_user)):
    """Get current user profile"""
    return {
        "success": True,
        "user": {
            "id": str(current_user["_id"]),
            "name": current_user["name"],
            "email": current_user["email"],
            "created_at": current_user["created_at"]
        }
    }

# WebSocket Endpoints
@app.websocket("/ws/terminal/{client_id}")
async def websocket_terminal(websocket: WebSocket, client_id: str):
    """WebSocket endpoint for interactive terminal"""
    await manager.connect(websocket, client_id)
    try:
        while True:
            data = await websocket.receive_text()
            command_data = json.loads(data)
            
            if command_data.get("type") == "command":
                command = command_data.get("command", "")
                output = await manager.execute_terminal_command(client_id, command)
                await manager.send_personal_message(
                    json.dumps({
                        "type": "output",
                        "output": output,
                        "command": command
                    }), 
                    websocket
                )
            elif command_data.get("type") == "resize":
                # Handle terminal resize (for future implementation)
                pass
                
    except WebSocketDisconnect:
        manager.disconnect(websocket, client_id)

@app.websocket("/ws/chat/{client_id}")
async def websocket_chat(websocket: WebSocket, client_id: str):
    """WebSocket endpoint for live chat streaming"""
    await manager.connect(websocket, client_id)
    chat_handler = ChatHandler()
    
    try:
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            if message_data.get("type") == "message":
                user_message = message_data.get("message", "")
                session_id = message_data.get("session_id", "")
                project_path = message_data.get("project_path")
                
                # Create chat request
                chat_request = ChatRequest(
                    message=user_message,
                    session_id=session_id
                )
                
                # Process the message
                response = await chat_handler.process_chat_request(chat_request, project_path)
                
                # Send response back via WebSocket
                await manager.send_personal_message(
                    json.dumps({
                        "type": "response",
                        "response": response
                    }), 
                    websocket
                )
                
    except WebSocketDisconnect:
        manager.disconnect(websocket, client_id)

# Enhanced Chat Endpoint with Project Context
@app.post("/api/chat/with-project")
async def chat_with_project_context(request: ChatWithProjectRequest, current_user: dict = Depends(get_current_user)):
    """Enhanced chat endpoint that can execute code changes based on user requests"""
    
    chat_handler = ChatHandler()
    user_id = str(current_user["_id"])
    
    # If session doesn't exist, create one
    if request.session_id not in AGENT_SESSIONS:
        AGENT_SESSIONS[request.session_id] = AdvancedAutonomousAgent(request.session_id)
    
    # If project path provided, set project context
    if request.project_path:
        full_path = os.path.join(WORKSPACE_DIR, request.project_path)
        if os.path.exists(full_path):
            agent = AGENT_SESSIONS[request.session_id]
            await agent.understand_project_structure(full_path)
    
    # Process the chat request
    chat_request = ChatRequest(
        message=request.message,
        session_id=request.session_id
    )
    
    result = await chat_handler.process_chat_request(chat_request, request.project_path, user_id)
    
    # Save conversation to database
    conversation_data = {
        "user_id": user_id,
        "session_id": request.session_id,
        "message": request.message,
        "response": result.get("response", ""),
        "execution_performed": result.get("execution_performed", False),
        "project_path": request.project_path,
        "created_at": datetime.utcnow()
    }
    await conversations_collection.insert_one(conversation_data)
    
    return result

# Enhanced main chat endpoint to handle both regular chat and code execution
@app.post("/api/chat")
async def chat_completion(request: ChatRequest, current_user: dict = Depends(get_current_user)):
    """Main chat endpoint that can handle both conversation and code execution"""
    
    user_id = str(current_user["_id"])
    
    # Check if this might be a code execution request
    chat_handler = ChatHandler()
    analysis = await chat_handler.analyze_user_intent(request.message)
    
    if analysis.get('requires_execution', False) and request.session_id:
        # Use the enhanced chat with project context
        chat_request = ChatWithProjectRequest(
            message=request.message,
            session_id=request.session_id,
            auto_execute=True
        )
        return await chat_with_project_context(chat_request)
    else:
        # Regular chat response
        response = await chat_handler.generate_chat_response(request)
        
        # Save conversation to database
        conversation_data = {
            "user_id": user_id,
            "session_id": request.session_id,
            "message": request.message,
            "response": response,
            "execution_performed": False,
            "created_at": datetime.utcnow()
        }
        await conversations_collection.insert_one(conversation_data)
        
        return {
            "success": True,
            "response": response,
            "session_id": request.session_id,
            "execution_performed": False
        }

# File Diff Endpoints
@app.post("/api/files/create-diff")
async def create_file_diff(request: FileDiffRequest, current_user: dict = Depends(get_current_user)):
    """Create a file diff preview"""
    user_id = str(current_user["_id"])
    request.user_id = user_id
    
    diff_response = await FileDiffManager.create_diff(request)
    return {
        "success": True,
        "diff": diff_response,
        "message": "File diff created successfully"
    }

@app.post("/api/files/apply-diff/{diff_id}")
async def apply_file_diff(diff_id: str, current_user: dict = Depends(get_current_user)):
    """Apply a file diff to update the file"""
    file_manager = FileManager()
    result = await file_manager.apply_diff(diff_id)
    return {
        "success": True,
        "result": result,
        "message": "File updated successfully"
    }

@app.get("/api/files/diffs/{session_id}")
async def get_session_diffs(session_id: str, limit: int = 50, current_user: dict = Depends(get_current_user)):
    """Get diffs for a session"""
    diffs = await FileDiffManager.get_diffs_by_session(session_id, limit)
    return {
        "success": True,
        "session_id": session_id,
        "diffs": diffs,
        "count": len(diffs)
    }

@app.get("/api/files/diff/{diff_id}")
async def get_diff(diff_id: str, current_user: dict = Depends(get_current_user)):
    """Get a specific diff by ID"""
    diff = await FileDiffManager.get_diff(diff_id)
    if diff:
        return {
            "success": True,
            "diff": diff
        }
    else:
        raise HTTPException(status_code=404, detail="Diff not found")

# Session Management Endpoints
@app.post("/api/sessions/create")
async def create_session(project_path: str = None, current_user: dict = Depends(get_current_user)):
    """Create a new session"""
    session_id = str(uuid.uuid4())
    user_id = str(current_user["_id"])
    
    # Create session in database
    session_data = {
        "user_id": user_id,
        "session_id": session_id,
        "project_path": project_path,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
        "is_active": True
    }
    await sessions_collection.insert_one(session_data)
    
    # Create agent session
    AGENT_SESSIONS[session_id] = AdvancedAutonomousAgent(session_id)
    
    # If project path provided, analyze it
    project_context = {}
    if project_path:
        full_path = os.path.join(WORKSPACE_DIR, project_path)
        if os.path.exists(full_path):
            agent = AGENT_SESSIONS[session_id]
            project_context = await agent.understand_project_structure(full_path)
    
    return {
        "success": True,
        "session_id": session_id,
        "project_context": project_context,
        "message": "Session created successfully"
    }

@app.get("/api/sessions/user-sessions")
async def get_user_sessions(current_user: dict = Depends(get_current_user)):
    """Get all sessions for the current user"""
    user_id = str(current_user["_id"])
    cursor = sessions_collection.find({"user_id": user_id}).sort("updated_at", -1)
    
    sessions = []
    async for session in cursor:
        sessions.append({
            "session_id": session["session_id"],
            "project_path": session.get("project_path"),
            "created_at": session["created_at"],
            "updated_at": session["updated_at"],
            "is_active": session.get("is_active", True)
        })
    
    return {
        "success": True,
        "sessions": sessions,
        "count": len(sessions)
    }

@app.get("/api/sessions/conversations/{session_id}")
async def get_session_conversations(session_id: str, current_user: dict = Depends(get_current_user)):
    """Get conversation history for a session"""
    user_id = str(current_user["_id"])
    cursor = conversations_collection.find({
        "user_id": user_id,
        "session_id": session_id
    }).sort("created_at", 1)
    
    conversations = []
    async for conv in cursor:
        conversations.append({
            "message": conv["message"],
            "response": conv["response"],
            "execution_performed": conv.get("execution_performed", False),
            "created_at": conv["created_at"]
        })
    
    return {
        "success": True,
        "session_id": session_id,
        "conversations": conversations,
        "count": len(conversations)
    }

# GitHub Repository Endpoints
@app.post("/api/github/clone")
async def clone_github_repository(request: GitHubCloneRequest, current_user: dict = Depends(get_current_user)):
    """Clone a GitHub repository to workspace"""
    github_manager = GitHubManager()
    result = await github_manager.clone_repository(
        request.repo_url, 
        request.target_dir, 
        request.session_id
    )
    return result

@app.get("/api/github/repo-info/{repo_path:path}")
async def get_repository_info(repo_path: str, current_user: dict = Depends(get_current_user)):
    """Get information about a cloned repository"""
    github_manager = GitHubManager()
    abs_path = os.path.join(WORKSPACE_DIR, repo_path)
    result = await github_manager.get_repo_info(abs_path)
    return result

# Enhanced Agent Endpoints
@app.post("/api/agent/create-session")
async def create_agent_session(project_path: str = None, current_user: dict = Depends(get_current_user)):
    """Create advanced agent session with optional project context"""
    session_id = str(uuid.uuid4())
    AGENT_SESSIONS[session_id] = AdvancedAutonomousAgent(session_id)
    
    # If project path provided, analyze it immediately
    project_context = {}
    if project_path:
        full_path = os.path.join(WORKSPACE_DIR, project_path)
        if os.path.exists(full_path):
            agent = AGENT_SESSIONS[session_id]
            project_context = await agent.understand_project_structure(full_path)
    
    return {
        "success": True, 
        "session_id": session_id, 
        "message": "Advanced agent session created",
        "project_context": project_context
    }

@app.post("/api/agent/set-project")
async def set_agent_project(session_id: str, project_path: str, current_user: dict = Depends(get_current_user)):
    """Set the project context for an agent session"""
    if not session_id or session_id not in AGENT_SESSIONS:
        raise HTTPException(status_code=404, detail="Agent session not found")
    
    agent = AGENT_SESSIONS[session_id]
    full_path = os.path.join(WORKSPACE_DIR, project_path)
    
    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail="Project path not found")
    
    project_context = await agent.understand_project_structure(full_path)
    
    return {
        "success": True,
        "session_id": session_id,
        "project_path": project_path,
        "project_context": project_context
    }

# MAIN IMPLEMENTATION ENDPOINT - LIKE CURSOR AI
@app.post("/api/agent/implement-requirements")
async def implement_project_requirements(request: ProjectRequirements, current_user: dict = Depends(get_current_user)):
    """Main endpoint for implementing project requirements like Cursor AI"""
    if not request.session_id or request.session_id not in AGENT_SESSIONS:
        raise HTTPException(status_code=404, detail="Agent session not found")
    
    agent = AGENT_SESSIONS[request.session_id]
    user_id = str(current_user["_id"])
    
    try:
        full_project_path = os.path.join(WORKSPACE_DIR, request.project_path)
        
        # Step 1: Understand project structure
        project_context = await agent.understand_project_structure(full_project_path)
        
        # Step 2: Generate implementation plan
        implementation_plan = await agent.generate_implementation_plan(
            request.requirements, 
            project_context
        )
        
        execution_results = None
        if request.auto_execute:
            # Step 3: Execute the plan - WITH DIFF PREVIEWS
            execution_results = await agent.execute_implementation_plan(implementation_plan, user_id)
        
        return {
            "success": True,
            "session_id": request.session_id,
            "project_path": request.project_path,
            "requirements": request.requirements,
            "project_analysis": project_context,
            "implementation_plan": implementation_plan,
            "execution_results": execution_results,
            "message": "Requirements processed successfully",
            "files_created": [step for step in implementation_plan.get('implementation_steps', []) 
                            if step.get('action') in ['create_file', 'created_file']],
            "files_updated": [step for step in implementation_plan.get('implementation_steps', []) 
                            if step.get('action') in ['update_file', 'updated_file']]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Requirements implementation failed: {str(e)}")

# SPECIALIZED E-COMMERCE ENDPOINT
@app.post("/api/agent/implement-ecommerce")
async def implement_ecommerce_features(
    session_id: str,
    project_path: str,
    requirements: str = "Convert this project into a dynamic e-commerce website with shopping cart, product listings, user authentication, and professional styling",
    current_user: dict = Depends(get_current_user)
):
    """Specialized endpoint for e-commerce implementations"""
    if not session_id or session_id not in AGENT_SESSIONS:
        raise HTTPException(status_code=404, detail="Agent session not found")
    
    agent = AGENT_SESSIONS[session_id]
    full_project_path = os.path.join(WORKSPACE_DIR, project_path)
    user_id = str(current_user["_id"])
    
    # Set project context
    await agent.understand_project_structure(full_project_path)
    
    # Implement e-commerce features
    result = await agent.implement_ecommerce_features(requirements, user_id)
    
    return {
        "success": True,
        "session_id": session_id,
        "project_path": project_path,
        "ecommerce_features": result["features_implemented"],
        "plan": result["ecommerce_plan"],
        "execution_results": result["execution_results"]
    }

# CSS UPDATE ENDPOINT
@app.post("/api/agent/update-css")
async def update_project_css(request: CSSUpdateRequest, current_user: dict = Depends(get_current_user)):
    """Specialized endpoint for CSS updates"""
    if not request.session_id or request.session_id not in AGENT_SESSIONS:
        raise HTTPException(status_code=404, detail="Agent session not found")
    
    agent = AGENT_SESSIONS[request.session_id]
    full_project_path = os.path.join(WORKSPACE_DIR, request.project_path)
    user_id = str(current_user["_id"])
    
    # Analyze current project structure
    project_context = await agent.understand_project_structure(full_project_path)
    
    # Generate professional CSS updates
    system_prompt = """You are a professional CSS/UI expert. Given the CSS requirements and project context, generate modern, professional CSS code.

    Project Context: {project_context}
    
    CSS Requirements: {css_requirements}
    Target Files: {target_files}
    
    Generate clean, modern CSS that follows best practices:
    - Use CSS variables for theming
    - Implement responsive design
    - Use flexbox/grid for layouts
    - Add smooth transitions
    - Ensure accessibility
    - Follow BEM naming convention if applicable
    
    Return JSON with:
    {{
        "analysis": "analysis of CSS needs",
        "css_updates": [
            {{
                "file_path": "path/to/css/file",
                "new_css": "professional css code here",
                "description": "what this CSS achieves"
            }}
        ],
        "recommendations": ["suggestion1", "suggestion2"]
    }}""".format(
        project_context=json.dumps(project_context, indent=2),
        css_requirements=request.css_requirements,
        target_files=request.target_files or ["global"]
    )
    
    contents = [{"role": "user", "parts": [{"text": f"Generate professional CSS for: {request.css_requirements}"}]}]
    payload = {"systemText": system_prompt, "contents": contents}
    
    response = await call_gemini_with_timeout(MODEL, payload)
    
    if "candidates" in response and len(response["candidates"]) > 0:
        candidate = response["candidates"][0]
        if "content" in candidate and "parts" in candidate["content"]:
            css_plan_text = candidate["content"]["parts"][0].get("text", "")
            
            try:
                start_idx = css_plan_text.find('{')
                end_idx = css_plan_text.rfind('}') + 1
                if start_idx != -1 and end_idx != -1:
                    css_plan = json.loads(css_plan_text[start_idx:end_idx])
                    
                    # Execute CSS updates - WITH DIFF PREVIEWS
                    results = []
                    for update in css_plan.get('css_updates', []):
                        result = await agent.update_css_styles(
                            update['file_path'],
                            update['new_css'],
                            update['description'],
                            user_id
                        )
                        results.append(result)
                    
                    return {
                        "success": True,
                        "css_plan": css_plan,
                        "execution_results": results,
                        "files_updated": len(results),
                        "message": f"Successfully updated {len(results)} CSS files"
                    }
            except json.JSONDecodeError:
                return {
                    "success": True,
                    "css_recommendations": css_plan_text,
                    "execution_results": []
                }
    
    raise HTTPException(status_code=500, detail="Failed to generate CSS updates")

# Quick Action Endpoints for Common Requests
@app.post("/api/quick/fix-bug")
async def quick_fix_bug(request: BugFixRequest, current_user: dict = Depends(get_current_user)):
    """Quick endpoint for bug fixes"""
    if not request.session_id or request.session_id not in AGENT_SESSIONS:
        raise HTTPException(status_code=404, detail="Agent session not found")
    
    agent = AGENT_SESSIONS[request.session_id]
    full_project_path = os.path.join(WORKSPACE_DIR, request.project_path)
    user_id = str(current_user["_id"])
    
    # Analyze and fix the bug
    system_prompt = f"""You are a debugging expert. Fix this bug:

    Error: {request.error_description}
    Code: {request.code_snippet}

    Provide the complete fixed code and explanation."""
    
    contents = [{"role": "user", "parts": [{"text": f"Fix this bug: {request.error_description}"}]}]
    payload = {"systemText": system_prompt, "contents": contents}
    
    response = await call_gemini_with_timeout(MODEL, payload)
    
    if "candidates" in response and len(response["candidates"]) > 0:
        candidate = response["candidates"][0]
        if "content" in candidate and "parts" in candidate["content"]:
            fix_solution = candidate["content"]["parts"][0].get("text", "")
            
            return {
                "success": True,
                "bug_description": request.error_description,
                "solution": fix_solution,
                "session_id": request.session_id
            }
    
    raise HTTPException(status_code=500, detail="Failed to generate bug fix")

@app.post("/api/quick/update-css")
async def quick_update_css(
    session_id: str,
    project_path: str,
    element: str,
    css_properties: str,
    current_user: dict = Depends(get_current_user)
):
    """Quick endpoint for CSS updates"""
    if not session_id or session_id not in AGENT_SESSIONS:
        raise HTTPException(status_code=404, detail="Agent session not found")
    
    agent = AGENT_SESSIONS[session_id]
    full_project_path = os.path.join(WORKSPACE_DIR, project_path)
    user_id = str(current_user["_id"])
    
    # Generate CSS code
    system_prompt = f"""Create CSS for this element: {element}
    Properties needed: {css_properties}
    
    Return just the CSS code without explanations."""
    
    contents = [{"role": "user", "parts": [{"text": f"CSS for {element}: {css_properties}"}]}]
    payload = {"systemText": system_prompt, "contents": contents}
    
    response = await call_gemini_with_timeout(MODEL, payload)
    
    if "candidates" in response and len(response["candidates"]) > 0:
        candidate = response["candidates"][0]
        if "content" in candidate and "parts" in candidate["content"]:
            css_code = candidate["content"]["parts"][0].get("text", "")
            
            # Apply to common CSS files with diff preview
            css_files = ["src/App.css", "src/index.css", "styles.css"]
            results = []
            
            for css_file in css_files:
                try:
                    result = await agent.update_css_styles(css_file, css_code, f"Update {element}", user_id)
                    results.append(result)
                except:
                    continue
            
            return {
                "success": True,
                "element": element,
                "css_code": css_code,
                "files_updated": [r for r in results if r.get('action') == 'css_updated'],
                "files_with_diff": [r for r in results if 'diff_id' in r],
                "session_id": session_id
            }
    
    raise HTTPException(status_code=500, detail="Failed to generate CSS")

# File Upload Endpoints
@app.post("/api/upload/file", response_model=UploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    session_id: str = Form(None),
    extract: bool = Form(False),
    current_user: dict = Depends(get_current_user)
):
    """Upload a single file"""
    upload_manager = UploadManager()
    file_info = await upload_manager.save_uploaded_file(file, session_id)
    
    result = {
        "filename": file_info["filename"],
        "path": file_info["path"],
        "size": file_info["size"]
    }
    
    # Extract if it's an archive and extraction is requested
    if extract and file_info["filename"].endswith(('.zip', '.tar', '.tar.gz', '.tgz')):
        extraction = await upload_manager.extract_archive(file_info["path"])
        result["file_count"] = extraction["file_count"]
        result["extracted_path"] = extraction["extracted_to"]
    
    return result

@app.post("/api/upload/folder")
async def upload_folder(
    files: List[UploadFile] = File(...),
    session_id: str = Form(None),
    current_user: dict = Depends(get_current_user)
):
    """Upload multiple files as a folder"""
    upload_manager = UploadManager()
    uploaded_files = []
    total_size = 0
    
    for file in files:
        file_info = await upload_manager.save_uploaded_file(file, session_id)
        uploaded_files.append({
            "filename": file_info["filename"],
            "size": file_info["size"]
        })
        total_size += file_info["size"]
    
    return {
        "success": True,
        "session_id": session_id,
        "file_count": len(uploaded_files),
        "total_size": total_size,
        "files": uploaded_files
    }

@app.post("/api/upload/copy-to-workspace")
async def copy_to_workspace(
    source_path: str,
    target_dir: str = None,
    current_user: dict = Depends(get_current_user)
):
    """Copy uploaded files to workspace"""
    upload_manager = UploadManager()
    result = await upload_manager.copy_to_workspace(source_path, target_dir)
    return result

# File Operations Endpoints
@app.post("/api/files/operation")
async def file_operation(operation: FileOperation, current_user: dict = Depends(get_current_user)):
    file_manager = FileManager()
    user_id = str(current_user["_id"])
    
    try:
        if operation.action == "create":
            result = await file_manager.create_file(operation.path, operation.content or "")
        elif operation.action == "read":
            result = await file_manager.read_file(operation.path)
        elif operation.action == "update":
            # Use update with diff for file updates
            result = await file_manager.update_file_with_diff(
                operation.path, 
                operation.content or "", 
                operation.session_id or "default",
                user_id
            )
        elif operation.action == "delete":
            result = await file_manager.delete_file(operation.path)
        elif operation.action == "list":
            result = await file_manager.list_files(operation.path or "")
        else:
            raise HTTPException(status_code=400, detail="Invalid file operation")
        
        return {"success": True, "operation": operation.action, "result": result}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Terminal Endpoints
@app.post("/api/terminal/execute")
async def execute_terminal_command(request: TerminalRequest, current_user: dict = Depends(get_current_user)):
    terminal = TerminalManager()
    result = await terminal.execute_command(request.command, request.working_dir or WORKSPACE_DIR)
    return {"success": True, "result": result}

# Workspace Management
@app.get("/api/workspace/info")
async def get_workspace_info(current_user: dict = Depends(get_current_user)):
    total_size = 0
    file_count = 0
    dir_count = 0
    projects = []
    
    for item in os.listdir(WORKSPACE_DIR):
        item_path = os.path.join(WORKSPACE_DIR, item)
        if os.path.isdir(item_path):
            projects.append(item)
            dir_count += 1
            for root, dirs, files in os.walk(item_path):
                dir_count += len(dirs)
                file_count += len(files)
                for file in files:
                    file_path = os.path.join(root, file)
                    total_size += os.path.getsize(file_path)
    
    return {
        "workspace_path": WORKSPACE_DIR,
        "total_size_bytes": total_size,
        "file_count": file_count,
        "directory_count": dir_count,
        "project_count": len(projects),
        "projects": projects,
        "active_sessions": len(AGENT_SESSIONS)
    }

if __name__ == "__app__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)