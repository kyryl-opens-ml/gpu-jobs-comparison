import runpod

runpod.api_key = "IOQA87IDZCITG92W82HJA34LWYT6MFJF2JY3FYYU"

# Get all my pods
pods = runpod.get_pods()

# Get a specific pod
pod = runpod.get_pod(pod.id)

# Create a pod
pod = runpod.create_pod("test", "runpod/stack", "NVIDIA GeForce RTX 3070")

# Stop the pod
runpod.stop_pod(pod.id)

# Resume the pod
runpod.resume_pod(pod.id)

# Terminate the pod
runpod.terminate_pod(pod.id)