import torch
import torchvision.models as models
import time

# Usa un modelo pre-entrenado (ResNet18)
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.eval()

# Crea un input aleatorio de tamaño típico (1 imagen, 3 canales, 224x224)
input_data = torch.randn(32, 3, 224, 224)

# -------- CPU --------
model_cpu = model.to("cpu")
input_cpu = input_data.to("cpu")

start_cpu = time.time()
with torch.no_grad():
    for _ in range(50):
        output_cpu = model_cpu(input_cpu)
end_cpu = time.time()

# -------- GPU (CUDA) --------
if torch.cuda.is_available():
    model_gpu = model.to("cuda")
    input_gpu = input_data.to("cuda")

    # Calienta la GPU (importante en benchmark)
    with torch.no_grad():
        for _ in range(5):
            _ = model_gpu(input_gpu)

    start_gpu = time.time()
    with torch.no_grad():
        for _ in range(50):
            output_gpu = model_gpu(input_gpu)
    end_gpu = time.time()
else:
    print("⚠️ CUDA no está disponible.")

# -------- Resultados --------
print(f"\n🔍 Inference time on CPU:  {end_cpu - start_cpu:.4f} s")
if torch.cuda.is_available():
    print(f"⚡ Inference time on GPU:  {end_gpu - start_gpu:.4f} s")
    speedup = (end_cpu - start_cpu) / (end_gpu - start_gpu)
    print(f"🚀 Speedup (CPU vs GPU):   x{speedup:.2f}")
