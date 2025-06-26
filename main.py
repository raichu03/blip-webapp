from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.templating import Jinja2Templates
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import io
import uvicorn

app = FastAPI(title="BLIP Image Processing API")

processor: BlipProcessor = None
model: BlipForConditionalGeneration = None

templates = Jinja2Templates(directory='.')

@app.on_event("startup")
async def load_model():
    """
    Load the BLIP model and processor once when the FastAPI application starts.
    This prevents reloading the model on every request, saving memory and time.
    """
    global processor, model
    try:
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    except Exception as e:
        raise RuntimeError("Failed to load BLIP model on startup.") from e

def process_image_file(file: UploadFile) -> Image.Image:
    """
    Helper function to read an uploaded image file and convert it to PIL Image format.
    """
    try:
        image_bytes = file.file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not process image file: {e}")

@app.get("/")
async def root(request: Request):
    context = {'request': request}
    return templates.TemplateResponse(
        name="index.html",
        context=context
    )

@app.post("/caption")
async def generate_image_caption_api(image: UploadFile = File(...)):
    """
    Generates a descriptive caption for an uploaded image.

    Args:
        image (UploadFile): The image file uploaded via the form.

    Returns:
        dict: A dictionary containing the generated caption.
    """
    if not processor or not model:
        raise HTTPException(status_code=503, detail="Model not loaded yet. Please wait or check server logs.")

    try:
        pil_image = process_image_file(image)

        inputs = processor(images=pil_image, return_tensors="pt")

        outputs = model.generate(**inputs, num_beams=4, max_length=50)

        caption = processor.decode(outputs[0], skip_special_tokens=True)
        return {"filename": image.filename, "caption": caption}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error during caption generation: {e}")

@app.post("/vqa")
async def answer_image_question_api(image: UploadFile = File(...), question: str = Form(...)):
    """
    Answers a question about an uploaded image (Visual Question Answering).

    Args:
        image (UploadFile): The image file uploaded via the form.
        question (str): The question related to the image, sent as form data.

    Returns:
        dict: A dictionary containing the question and the generated answer.
    """
    if not processor or not model:
        raise HTTPException(status_code=503, detail="Model not loaded yet. Please wait or check server logs.")

    try:
        pil_image = process_image_file(image)

        inputs = processor(images=pil_image, text=question, return_tensors="pt")

        outputs = model.generate(**inputs, num_beams=4)

        answer = processor.decode(outputs[0], skip_special_tokens=True)
        return {"answer": answer}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error during VQA: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
