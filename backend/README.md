# CliniqAI Backend -- Stage 1 (Basic Analysis Pipeline)

Single endpoint: upload an image, get back gatekeeper -> modality -> specialist diagnosis.
No auth, no database, no history yet -- that's the deliberately deferred next phase.

## Setup

1. Copy this whole `backend/` folder into your project, e.g. as
   `E:\programming\CliniqAI\backend\`.

2. Copy your actual trained checkpoints into the matching folders here:
   - `backend/models/router/router_stage1_efficientnet_b0.pt`
   - `backend/models/router/router_stage2_efficientnet_b0.pt`
   - `backend/models/specialists/*.pt` (all 7 specialist checkpoints)

   These folders are currently empty placeholders -- the app won't start without
   the real files in place.

3. Install dependencies (ideally in your existing `CliniqAI` conda env, or a
   fresh one for the backend specifically):
   ```
   pip install -r requirements.txt
   ```

4. Run it:
   ```
   uvicorn app.main:app --reload --port 8000
   ```

5. Watch the startup logs carefully. For each of the 9 models you'll see either:
   ```
   [loaded] specialist:brain_mri: 4 classes -> ['glioma', 'meningioma', 'notumor', 'pituitary']
   ```
   or a `[warning]` if a checkpoint is missing `class_names` (placeholder labels
   used instead -- fix this before trusting that model's output labels) or if a
   checkpoint file is missing entirely (that modality will fail at request time).

   **Read every line of this startup output once, the first time you run it.**
   This is where you'll discover which of your 7 specialists need their
   `class_names` checked or fixed, since 4 of them weren't built in a session I
   had visibility into and I genuinely don't know their exact checkpoint format.

## Testing

Health check:
```
curl http://localhost:8000/health
```

Should list all specialists that loaded successfully.

Run an actual image through the pipeline:
```
curl -X POST http://localhost:8000/analyze -F "file=@path/to/test_image.jpg"
```

Test with:
- A real image from each of your 7 specialists' test sets (confirm correct routing)
- A random non-medical photo (confirm Stage 1 rejects it with `is_medical: false`)
- A corrupted/truncated image file (confirm it returns a clean 400 error, not a crash)

## Response shape

```json
{
  "is_medical": true,
  "gatekeeper_confidence": 0.999,
  "modality": "retinal",
  "modality_confidence": 0.97,
  "prediction": "moderate_dr",
  "prediction_confidence": 0.82,
  "all_class_probabilities": {"no_dr": 0.02, "mild_dr": 0.05, "moderate_dr": 0.82, ...},
  "low_confidence_warning": false,
  "message": "OK"
}
```

Same shape regardless of which specialist handled it -- your frontend doesn't need
per-modality branching, just render `prediction` + `prediction_confidence` generically,
and use `all_class_probabilities` if you want a breakdown chart.

## Known gaps (deliberately out of scope for this phase)

- No auth, no user accounts, no MongoDB, no request history
- No Grad-CAM explainability
- No Google Cloud Storage integration -- models load from local disk / the
  Docker image itself
- `CORSMiddleware` is wide open (`allow_origins=["*"]`) -- restrict before any
  public deployment

## Deploying to Hugging Face Spaces (when ready)

1. Create a new Space, SDK: Docker
2. Push this whole folder (including populated `models/`) via git -- use
   `git lfs track "*.pt"` first, since the combined checkpoints are ~150-200MB
3. The Dockerfile here already listens on port 7860, which is what Spaces expects
