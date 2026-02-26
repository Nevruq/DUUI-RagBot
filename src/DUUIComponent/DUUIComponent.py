import os
import sys

# Add directories to path for imports
_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_current_dir)

if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

import llm_wrapper
import load_hf_model



def generate_duui_component(
    model_id: str,
    component_name: str,
    output_path: str,
    verbose: bool = True
) -> dict:
    """
    Generates a complete DUUI component from a HuggingFace model.

    Args:
        model_id: HuggingFace model ID (e.g., "debajyotimaz/codemix_hate")
        component_name: Name for the component (e.g., "hate", "nsfw", "sentiment")
        output_path: Path where the component folder will be created
        verbose: Print progress messages

    Returns:
        dict with paths to all generated files

    Generated files:
        - TypeSystem.xml
        - duui_{component_name}.lua
        - duui_{component_name}.py
        - Dockerfile
        - docker_build.sh
    """

    # Create output directory
    component_dir = os.path.join(output_path, f"duui-{component_name}")
    os.makedirs(component_dir, exist_ok=True)

    if verbose:
        print(f"=== DUUI Component Generator ===")
        print(f"Model ID: {model_id}")
        print(f"Component Name: {component_name}")
        print(f"Output Directory: {component_dir}")
        print()

    # Initialize LLM wrapper
    llm = llm_wrapper.LLMWrapper()

    # Step 1: Get HuggingFace model information
    if verbose:
        print("1. Fetching HuggingFace model information...")
    hf_model_info = load_hf_model.inspect_hf_model(model_id=model_id)
    if verbose:
        print(f"   Task: {hf_model_info.get('inferred_task', 'unknown')}")
        print(f"   Labels: {hf_model_info.get('id2label', {})}")
        print()

    # Step 2: Generate TypeSystem
    if verbose:
        print("2. Generating TypeSystem.xml...")
    typesystem_xml = llm.llm_typesystem_builder(hf_model_json=hf_model_info)
    typesystem_path = os.path.join(component_dir, "TypeSystem.xml")
    with open(typesystem_path, 'w', encoding='utf-8') as f:
        f.write(typesystem_xml)
    if verbose:
        print(f"   Saved: {typesystem_path}")
        print()

    # Step 3: Generate Lua script
    if verbose:
        print(f"3. Generating duui_{component_name}.lua...")
    lua_script = llm.llm_lua_code_builder(
        hf_model_json=hf_model_info,
        typesystem_xml=typesystem_xml
    )
    lua_path = os.path.join(component_dir, f"duui_{component_name}.lua")
    with open(lua_path, 'w', encoding='utf-8') as f:
        f.write(lua_script)
    if verbose:
        print(f"   Saved: {lua_path}")
        print()

    # Step 4: Generate Python script
    if verbose:
        print(f"4. Generating duui_{component_name}.py...")
    python_script = llm.llm_python_code_builder(
        hf_model_json=hf_model_info,
        component_name=component_name
    )
    python_path = os.path.join(component_dir, f"duui_{component_name}.py")
    with open(python_path, 'w', encoding='utf-8') as f:
        f.write(python_script)
    if verbose:
        print(f"   Saved: {python_path}")
        print()

    # Step 5: Generate Dockerfile
    if verbose:
        print("5. Generating Dockerfile...")
    dockerfile = llm.llm_dockerfile_builder(
        hf_model_json=hf_model_info,
        component_name=component_name
    )
    dockerfile_path = os.path.join(component_dir, "Dockerfile")
    with open(dockerfile_path, 'w', encoding='utf-8') as f:
        f.write(dockerfile)
    if verbose:
        print(f"   Saved: {dockerfile_path}")
        print()

    # Step 6: Generate docker_build.sh
    if verbose:
        print("6. Generating docker_build.sh...")
    docker_build = llm.llm_docker_build_builder(
        hf_model_json=hf_model_info,
        component_name=component_name
    )
    docker_build_path = os.path.join(component_dir, "docker_build.sh")
    with open(docker_build_path, 'w', encoding='utf-8') as f:
        f.write(docker_build)
    # Make executable
    os.chmod(docker_build_path, 0o755)
    if verbose:
        print(f"   Saved: {docker_build_path}")
        print()

    # Step 7: Generate requirements.txt (static)
    if verbose:
        print("7. Generating requirements.txt...")
    requirements = """transformers>=4.40.0
                        torch>=2.0.0
                        fastapi>=0.110.0
                        uvicorn[standard]>=0.27.0
                        dkpro-cassis>=0.9.0
                        pydantic>=2.0.0
                        """
    requirements_path = os.path.join(component_dir, "requirements.txt")
    with open(requirements_path, 'w', encoding='utf-8') as f:
        f.write(requirements)
    if verbose:
        print(f"   Saved: {requirements_path}")
        print()

    # Summary
    generated_files = {
        "component_dir": component_dir,
        "typesystem": typesystem_path,
        "lua_script": lua_path,
        "python_script": python_path,
        "dockerfile": dockerfile_path,
        "docker_build": docker_build_path,
        "requirements": requirements_path,
    }

    if verbose:
        print("=== Generation Complete ===")
        print(f"Component directory: {component_dir}")
        print()
        print("Files generated:")
        print(f"  - TypeSystem.xml")
        print(f"  - duui_{component_name}.lua")
        print(f"  - duui_{component_name}.py")
        print(f"  - Dockerfile")
        print(f"  - docker_build.sh")
        print(f"  - requirements.txt")
        print()
        print("Next steps:")
        print(f"  1. cd {component_dir}")
        print(f"  2. ./docker_build.sh")
        print(f"  3. docker run -p 9714:9714 duui-{component_name}:latest")

    return generated_files


if __name__ == "__main__":
    pass
