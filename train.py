import torch
import numpy as np
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_TESSELATION
from models.combined import FaceCursorModel

EPOCHS = 8
LR = 1e-3
MODEL_PATH = "models/face_cursor.pth"


raw = np.load("data/samples.npz", allow_pickle=True)["data"]

data = []
for i in range(1, len(raw)):
    prev_landmarks, prev_nose = raw[i - 1]
    landmarks, nose = raw[i]

    dx = nose[0] - prev_nose[0]
    dy = nose[1] - prev_nose[1]

    data.append((landmarks, [dx, dy]))

print(f"Training on {len(data)} velocity samples")


edges = torch.tensor(
    [[a, b] for a, b in FACEMESH_TESSELATION],
    dtype=torch.long
).t().contiguous()


model = FaceCursorModel()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
best_loss = float("inf")


try:
    for epoch in range(EPOCHS):
        total_loss = 0.0

        for landmarks, vel in data:
            x = torch.tensor(landmarks, dtype=torch.float)
            y = torch.tensor(vel, dtype=torch.float)

            pred = model(x, edges)
            loss = ((pred - y) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg = total_loss / len(data)
        print(f"Epoch {epoch+1}, Loss {avg:.6f}")

        if avg < best_loss:
            best_loss = avg
            torch.save(model.state_dict(), MODEL_PATH)
            print("  ↳ Best model saved")

except KeyboardInterrupt:
    print("Training interrupted")

finally:
    torch.save(model.state_dict(), MODEL_PATH)
    print("Final model saved")
