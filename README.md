# 🚀 CUDA Vector Addition

A simple CUDA project implemented by **Prajwal Mededar** to demonstrate **parallel computing** using GPU acceleration.  
This program performs **vector addition (C = A + B)** on the GPU using **Numba and CUDA**.

---

## 📘 Objective
To utilize GPU parallelism by implementing vector addition using CUDA — where each thread handles the addition of one element in the vector.

---

## ⚙️ Libraries Used
- **numba** → To write and execute CUDA kernels in Python  
- **numpy** → For numerical array creation and manipulation

---

## 🧩 Implementation Details

### 1️⃣ Defining the CUDA Kernel
- The kernel is written using `@cuda.jit`, which compiles it for CUDA-enabled GPUs.  
- `cuda.grid(1)` gives the **global thread index**, ensuring each thread processes one element.  
- Each thread performs:  
  ```python
  c[idx] = a[idx] + b[idx]

2️⃣ Main Program Steps

Vector size: 1024 elements

Initialize:

a = np.ones(n, dtype=np.float32)
b = np.ones(n, dtype=np.float32)
c = np.zeros(n, dtype=np.float32)


Transfer data to GPU using cuda.to_device()

Configure parallel execution:

Threads per block = 256  
Blocks per grid = 4  
(4 × 256 = 1024 threads total)


Launch the CUDA kernel:

vector_addition[blocks_per_grid, threads_per_block](a_device, b_device, c_device)


Copy result back to CPU memory:

c = c_device.copy_to_host()


Print output:

Result of vector addition: [2. 2. 2. ... 2.]

🧠 Learning Outcome

Understanding of thread indexing (cuda.grid(1))

Host ↔ Device memory transfer

Launching and managing GPU kernels

Demonstration of GPU utilization in basic numerical operations

🖥️ How to Run

Install dependencies:

pip install numba numpy


Save the code as vector_addition.py

Run the file:

python vector_addition.py


Ensure your system has an NVIDIA GPU with CUDA support.