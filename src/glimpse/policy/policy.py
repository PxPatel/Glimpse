import json
import inspect
from pathlib import Path
from typing import Optional

class TracingPolicy:
    def __init__(self, included_packages=None, trace_depth=5, project_root_packages=None, auto_trace_subpackages=False):
        self.included_packages = set(included_packages or [])
        self.project_root_packages = set(project_root_packages or [])
        self.trace_depth = trace_depth 
        self.auto_trace_subpackages = auto_trace_subpackages if self.project_root_packages is not None and isinstance(self.project_root_packages, list) and len(self.project_root_packages) else False

    @classmethod
    def load(cls, policy_path: Optional[str] = None):
        """
        Load TracingPolicy from JSON file.
        
        Args:
            policy_path: Explicit path to policy file. Must be named 'glimpse-policy.json'
                        If None, searches for closest 'glimpse-policy.json' from caller location
        
        Returns:
            TracingPolicy instance
            
        Raises:
            FileNotFoundError: If policy file not found
            ValueError: If provided path doesn't end with 'glimpse-policy.json'
            JSONDecodeError: If policy file has invalid JSON
        """
        
        if policy_path is not None:
            # Case 1: Explicit path provided
            policy_file = Path(policy_path)
            
            # Validate filename
            if policy_file.name != "glimpse-policy.json":
                raise ValueError(
                    f"Policy file must be named 'glimpse-policy.json', got '{policy_file.name}'"
                )
            
            # Check if file exists
            if not policy_file.exists():
                raise FileNotFoundError(f"Policy file not found: {policy_path}")
                
            target_file = policy_file
            
        else:
            # Case 2: Discover closest policy file from caller location
            caller_frame = inspect.currentframe().f_back
            caller_file = Path(caller_frame.f_code.co_filename).resolve()
            
            # Walk up directory tree from caller's location
            search_dir = caller_file.parent
            target_file = None
            
            while search_dir != search_dir.parent:  # Stop at filesystem root
                candidate = search_dir / "glimpse-policy.json"
                if candidate.exists():
                    target_file = candidate
                    break
                search_dir = search_dir.parent
            
            if target_file is None:
                raise FileNotFoundError(
                    f"No 'glimpse-policy.json' found in directory tree starting from {caller_file.parent}"
                )
        
        # Load and parse JSON
        try:
            with open(target_file, 'r', encoding='utf-8') as f:
                policy_data = json.load(f)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"Invalid JSON in policy file {target_file}: {e.msg}",
                e.doc,
                e.pos
            )
        
        # Extract configuration with defaults
        included_packages = policy_data.get("included_packages", [])
        trace_depth = policy_data.get("trace_depth", 5)
        auto_trace_subpackages = policy_data.get("auto_trace_subpackages", True)
        project_root_packages = policy_data.get("project_root_packages", None)
        
        return cls(
            included_packages=included_packages,
            trace_depth=trace_depth,
            auto_trace_subpackages=auto_trace_subpackages,
            project_root_packages=project_root_packages
        )

    def should_trace_package(self, module_name: str) -> bool:
        """Check if a package should be traced based on policy rules."""
        if self.auto_trace_subpackages:
            for root in self.project_root_packages:
                if module_name.startswith(root):
                    return True
        
        if not self.included_packages:
            return False
            
        for package in self.included_packages:
            if module_name.startswith(package):
                return True

        return False