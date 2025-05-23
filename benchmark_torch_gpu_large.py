import torch
import torchvision.models as models
import time

# ⚙️ Configuraciones
BATCH_SIZE = 64
ITERATIONS = 200

print(f"🔧 Batch size: {BATCH_SIZE}, Iterations: {ITERATIONS}")

# 🧠 Modelo más grande
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.eval()

# Input de prueba
input_data = torch.randn(BATCH_SIZE, 3, 224, 224)

# -------- CPU --------
model_cpu = model.to("cpu")
input_cpu = input_data.to("cpu")

start_cpu = time.time()
with torch.no_grad():
    for _ in range(ITERATIONS):
        _ = model_cpu(input_cpu)
end_cpu = time.time()

# -------- GPU --------
if torch.cuda.is_available():
    model_gpu = model.to("cuda")
    input_gpu = input_data.to("cuda")

    # Warm-up
    with torch.no_grad():
        for _ in range(5):
            _ = model_gpu(input_gpu)

    start_gpu = time.time()
    with torch.no_grad():
        for _ in range(ITERATIONS):
            _ = model_gpu(input_gpu)
    end_gpu = time.time()
else:
    print("⚠️ CUDA no está disponible.")

# -------- Resultados --------
print(f"\n🔍 Inference time on CPU:  {end_cpu - start_cpu:.2f} s")
if torch.cuda.is_available():
    print(f"⚡ Inference time on GPU:  {end_gpu - start_gpu:.2f} s")
    speedup = (end_cpu - start_cpu) / (end_gpu - start_gpu)
    print(f"🚀 Speedup (CPU vs GPU):   x{speedup:.2f}")
