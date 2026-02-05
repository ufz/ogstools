"""
Saving and Loading Models - The Storage System
===============================================

OGSTools provides a unified storage system for managing simulation artifacts.

This guide uses :py:class:`~ogstools.Model` for all examples, but the same
principles and methods apply to all storage-capable objects. All inherit from
:py:class:`~ogstools.core.storage.StorageBase` and share the same storage interface.


The examples start with basic operations and progress to advanced features.
"""

# %%
from pathlib import Path
from tempfile import mkdtemp

import ogstools as ot
from ogstools.definitions import EXAMPLES_DIR
from ogstools.gmsh_tools import rect

# Set up a temporary directory for our examples
work_dir = Path(mkdtemp())
ot.StorageBase.Userpath = work_dir

# %%
# Basic Usage: Creating and Saving a Model
# -----------------------------------------
# Create and save a model in just a few lines.

# Create components
meshes = ot.Meshes.from_gmsh(rect(n_edge_cells=10))
prj_path = EXAMPLES_DIR / "prj" / "mechanics.prj"
project = ot.Project(input_file=prj_path)

# Create and save the model
my_model = ot.Model(project=project, meshes=meshes)
# Provide a path OR just .save()
my_model.save(work_dir / "my_model")

print(f"Model saved to: {my_model.active_target}")

# %%
# Loading a Model
# ---------------
# Load a previously saved model using its path.

loaded_model = ot.Model.from_folder(work_dir / "my_model")
print(loaded_model)

# %%
# Using IDs for Organization
# ---------------------------
# Alternatively, assign an ID to organize models in class-specific subdirectories.
# All objects with assigned id will be stored under ot.StorageBase.Userpath
model_experiment_1 = ot.Model(project=project, meshes=meshes, id="experiment_1")
model_experiment_1.save()
print(f"Model saved to: {model_experiment_1.active_target}")

# Load by ID
model_copy_experiment_1 = ot.Model.from_id("experiment_1")
print(f"Loaded by ID: {model_copy_experiment_1.active_target}")

# %%
# Overwriting with Backup
# -----------------------
# By default, overwriting requires explicit permission and creates backups.
# Change the behavior by changing ot.StorageBase.Backup=True/False

my_model.save(work_dir / "my_first_model", overwrite=True)

# %%
# Archiving for Sharing
# ---------------------
# Use archive=True to create self-contained copies (replaces symlinks with files).

my_model.save(work_dir / "archived", archive=True, overwrite=True)

# %%
# Roundtrip: Save and Load
# ------------------------
# All storage classes support complete roundtrip (save → load → identical object).
# The __repr__ output shows how to reload objects:

print(my_model)
# Output shows: Model.from_folder('/path/to/repr_example')

# %%
# ============================================================================
# Advanced Topics
# ============================================================================
# The following sections cover advanced storage concepts and internal behavior.

# %%
# Storage Principles Overview
# ---------------------------
# The storage system follows these core principles:
#
# - **Object-centric design**: Work with Python objects; files are technical artifacts
# - **Never overwrites originals**: file → (copy) → object → new file (by default)
# - **Cascading storage**: Saving a parent (Model) automatically saves children
# - **Deepcopy-aware**: Copied objects get new, independent save targets
# - **UserPath**: Set ``StorageBase.Userpath`` for organized, ID-based storage

# %%
# Inspecting Save Status
# ----------------------
# Objects track their save state through several properties.

model_status = ot.Model(project=project, meshes=meshes)
print("Before saving:")
print(f"  is_saved: {model_status.is_saved}")
print(f"  active_target: {model_status.active_target}")
print(f"  next_target: {model_status.next_target}")

model_status.save(work_dir / "status_example")
print("\nAfter saving:")
print(f"  is_saved: {model_status.is_saved}")
print(f"  active_target: {model_status.active_target}")

# %%
# Object-Centric Design: Files as Technical Artifacts
# ---------------------------------------------------
# OGSTools follows an object-centric philosophy: you work with Model objects,
# and files are just technical artifacts created when needed.

# Load and work with objects in memory
model_obj = ot.Model(project=ot.Project(input_file=prj_path), meshes=meshes)
# Files are created only when you explicitly save
model_obj.save(work_dir / "model_artifact")
print(f"Files created at: {model_obj.active_target}")

# %%
# Cascading Storage: Hierarchical Save Propagation
# ------------------------------------------------
# Saving a Model automatically saves all children (Project, Meshes, Execution).

model_cascade = ot.Model(
    project=ot.Project(input_file=prj_path),
    meshes=ot.Meshes.from_gmsh(rect(n_edge_cells=5)),
)
print(f"Before save - Meshes is_saved: {model_cascade.meshes.is_saved}")
print(f"Before save - Project is_saved: {model_cascade.project.is_saved}")

model_cascade.save(work_dir / "cascade_model")

print(f"After save - Meshes is_saved: {model_cascade.meshes.is_saved}")
print(f"After save - Project is_saved: {model_cascade.project.is_saved}")
print("All components saved under the Model folder!")

# Show the folder structure
print("\nFolder structure created:")
for item in sorted((work_dir / "cascade_model").rglob("*")):
    if item.is_file():
        rel_path = item.relative_to(work_dir / "cascade_model")
        print(f"  {rel_path}")

# %%
# Control Child Storage: Pre-specifying Child Targets
# ---------------------------------------------------
# You can control where children are saved by specifying their targets BEFORE
# saving the parent. This is useful for sharing components between models.

# Create meshes and save to a shared location first
shared_meshes = ot.Meshes.from_gmsh(rect(n_edge_cells=6))
shared_meshes.save(work_dir / "shared_meshes")
print(f"Meshes saved to: {shared_meshes.active_target}")

# Create a model with the pre-saved meshes
model_controlled = ot.Model(
    project=ot.Project(input_file=prj_path), meshes=shared_meshes
)
print("\nBefore Model.save():")
print(f"  Meshes user_specified_target: {shared_meshes.user_specified_target}")

# Save the model - meshes location is respected, not moved!
# It would be fine to decide here (short before model.save()) where to store the children (e.g. meshes)
model_controlled.save(work_dir / "controlled_model")

print("\nAfter Model.save():")
print(f"  Model saved to: {model_controlled.active_target}")
print(f"  Meshes still at: {model_controlled.meshes.active_target}")
print("  Meshes NOT moved into Model folder - pre-specified target respected!")

# Show both folder structures to illustrate the separation
print("\nModel folder structure:")
for item in sorted((work_dir / "controlled_model").rglob("*")):
    if item.is_file():
        rel_path = item.relative_to(work_dir / "controlled_model")
        print(f"  controlled_model/{rel_path}")

print("\nShared meshes folder (separate location):")
for item in sorted((work_dir / "shared_meshes").rglob("*")):
    if item.is_file():
        rel_path = item.relative_to(work_dir / "shared_meshes")
        print(f"  shared_meshes/{rel_path}")

# %%
# Deepcopy-Aware: Independent Storage for Copies
# ----------------------------------------------
# Deepcopy creates an independent Model with a new save target.

from copy import deepcopy

original = ot.Model(project=project, meshes=meshes)
original.save(work_dir / "original_model")

copied = deepcopy(original)
print(f"Copy is_saved: {copied.is_saved}")  # False - needs its own save
copied.save(work_dir / "copied_model")
print(f"Original: {original.active_target}")
print(f"Copy: {copied.active_target}")

# Show both folders exist independently
print("\nBoth models exist independently:")
print(f"  original_model/ exists: {(work_dir / 'original_model').exists()}")
print(f"  copied_model/ exists: {(work_dir / 'copied_model').exists()}")

# %%
# UserPath and Organized Storage
# -------------------------------
# Setting UserPath creates organized, class-specific subdirectories for IDs.

ot.StorageBase.Userpath = work_dir / "organized"
# Do this always in the beginning and do not change it in the middle of your workflow.
# It is recommended to have a variable name similar to the id.
model_experiment_A = ot.Model(project=project, meshes=meshes, id="experiment_A")
model_experiment_A.save()
print(f"Saved to: {model_experiment_A.active_target}")

# Show the organized folder structure with class-specific subdirectories
print("\nUserPath folder structure:")
for item in sorted((work_dir / "organized").rglob("*")):
    rel_path = item.relative_to(work_dir / "organized")
    indent = "  " * (len(rel_path.parts) - 1)
    if item.is_dir():
        print(f"{indent}{rel_path.name}/")
    else:
        print(f"{indent}{rel_path.name}")


# %% [markdown]
# ## Roundtrip Methods
#
# **Saving and loading methods for each class:**
#
# - **ogstools.Model**: ``.save()`` → ``Model.from_folder(path)`` / ``Model.from_id(id)``
# - **ogstools.Simulation**: ``.save()`` → ``Simulation.from_folder(path)`` / ``Simulation.from_id(id)``
# - **ogstools.Meshes**: ``.save()`` → ``Meshes.from_folder(path)`` / ``Meshes.from_file(meta)``
# - **ogstools.Project**: ``.save()`` → ``Project(input_file=path)``
# - **ogstools.MeshSeries**: ``.save()`` → ``MeshSeries(filepath=path)``
# - **ogstools.Execution**: ``.save()`` → ``Execution.from_file(path)``
