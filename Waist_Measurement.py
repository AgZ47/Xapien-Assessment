import trimesh
import numpy as np

def calculate_circumference(mesh, height_y): # Slices the mesh at certain height using a plane and calculates the circumference
    # Define the cutting plane (Adjusted for Y-Up)
    plane_origin = [0, height_y, 0] 
    plane_normal = [0, 1, 0] # Pointing up the Y-axis

    # Calculate the intersection 
    slice_path = mesh.section(plane_origin=plane_origin, plane_normal=plane_normal)

    if slice_path is None:
        print("Error: The plane did not intersect the mesh at this height.")
        return 0.0, None

    # Isolate the torso
    max_length = 0
    waist_points = None

    for curve_points in slice_path.discrete:
        segment_lengths = np.linalg.norm(np.diff(curve_points, axis=0), axis=1)
        curve_length = np.sum(segment_lengths)
        
        if curve_length > max_length:
            max_length = curve_length
            waist_points = curve_points
            
    if waist_points is not None:
        waist_slice = trimesh.load_path(waist_points)
        return max_length, waist_slice
        
    return 0.0, slice_path

if __name__ == "__main__":
    try:
        mesh = trimesh.load('image_smpl.obj', force='mesh')
    except Exception as e:
        print(f"Failed to load mesh: {e}")
        exit()

    # mesh.bounds returns [[xmin, ymin, zmin], [xmax, ymax, zmax]]
    # We now target index 1 for the Y-axis
    y_min = mesh.bounds[0][1] 
    y_max = mesh.bounds[1][1]
    print(f"Mesh Bounds - Feet: {y_min:.2f}, Head: {y_max:.2f}")

    target_ratio = 0.6 
    target_y = y_min + target_ratio * (y_max - y_min)
    print(f"Slicing at Y-Height: {target_y:.2f}")

    circumference, waist_slice = calculate_circumference(mesh, target_y)
    print(f"Calculated Waist Circumference: {circumference:.2f} units")

    if waist_slice is not None:
        mesh.visual.face_colors = [200, 200, 200, 100] 
        waist_slice.colors = [[255, 0, 0, 255]] 
        
        scene = trimesh.Scene([mesh, waist_slice])
        print("Opening visualization window...")
        scene.show()
