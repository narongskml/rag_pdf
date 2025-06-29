import base64
import gradio as gr
import os
import shutil
import pymupdf as fitz  # PyMuPDF
import numpy as np
from PIL import Image
import io
import chromadb
from chromadb.config import Settings
from transformers import CLIPProcessor, CLIPModel
import ollama
from typing import List, Dict, Tuple
import shortuuid
from scipy.spatial.distance import cosine
import logging
import re

TEMP_IMG="./data/images"
CHAT_MODEL ="qwen2.5vl:3b"

# CHAT_MODEL ="qwen2.5vl:3b", gemma3:4b,  granite3.2-vision:latest
# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize models and 
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Chroma client with telemetry disabled
chroma_client = chromadb.PersistentClient(path="./chroma_db_thai", settings=Settings(anonymized_telemetry=False))
collection = chroma_client.get_or_create_collection(name="pdf_data_thai")

# Create directory for storing images

os.makedirs(TEMP_IMG, exist_ok=True)


def extract_pdf_content(pdf_path: str) -> List[Dict]:
    """
    แยกข้อความและรูปภาพจาก PDF โดยใช้ PyMuPDF
    """
    try:
        doc = fitz.open(pdf_path)
        content_chunks = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            # Extract text
            text = page.get_text("text").strip()
            if not text:
                text = f"ไม่มีข้อความในหน้า {page_num + 1}"
            
            logging.info("################# Text data ##################")
            chunk_data = {"text": text, "images": []}
            
            # Extract images
            image_list = page.get_images(full=True)
            logging.info("################# images list ##################")
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                
                # Convert to PIL Image
                try:
                    image = Image.open(io.BytesIO(image_bytes))
                    if image.mode != "RGB":
                        image = image.convert("RGB")
                    
                    img_id = shortuuid.uuid()[:8]
                    img_path = f"{TEMP_IMG}/{img_id}.{image_ext}"
                    image.save(img_path, format=image_ext.upper())

                    img_desc = f"รูปภาพจากหน้า {page_num + 1} ของ PDF, บริบทข้อความ: {text[:100]}..."  
                    chunk_data["text"] += f"\n[ภาพ: {img_id}.{image_ext}]"                    
                    chunk_data["images"].append({
                        "data": image,
                        "path": img_path,
                        "description": img_desc
                    })
                except Exception as e:
                    logger.warning(f"ไม่สามารถประมวลผลรูปภาพที่หน้า {page_num + 1}, ดัชนี {img_index}: {str(e)}")
            
            if chunk_data["text"]:
                content_chunks.append(chunk_data)
        
        if not any(chunk["images"] for chunk in content_chunks):
            logger.warning("ไม่พบรูปภาพใน PDF: %s", pdf_path)
        
        doc.close()
        return content_chunks
    except Exception as e:
        logger.error("เกิดข้อผิดพลาดในการแยก PDF: %s", str(e))
        raise

def embed_text(text: str) -> np.ndarray:
    """
    สร้าง embedding สำหรับข้อความโดยใช้ CLIP
    """
    inputs = clip_processor(text=[text], return_tensors="pt", padding=True, truncation=True)
    outputs = clip_model.get_text_features(**inputs)
    return outputs.detach().numpy()[0]

def embed_image(image: Image.Image) -> np.ndarray:
    """
    สร้าง embedding สำหรับรูปภาพโดยใช้ CLIP
    """
    inputs = clip_processor(images=image, return_tensors="pt")
    outputs = clip_model.get_image_features(**inputs)
    return outputs.detach().numpy()[0]

def is_image_relevant(image: Image.Image, description: str, threshold: float = 0.7) -> bool:
    """
    ตรวจสอบว่ารูปภาพเกี่ยวข้องกับบริบทหรือไม่โดยใช้ CLIP
    """
    if not description:
        return False
    text_embedding = embed_text(description)
    image_embedding = embed_image(image)
    similarity = 1 - cosine(text_embedding, image_embedding)
    return similarity >= threshold

def store_in_chroma(content_chunks: List[Dict], pdf_name: str):
    """
    เก็บข้อมูลข้อความและรูปภาพใน Chroma พร้อม embedding
    """
    logging.info("##### Start store in chroma #########")
    for chunk in content_chunks:
        text = chunk["text"]
        images = chunk["images"]
        logging.info("################# Text embeding store ##################")
        text_embedding = embed_text(text)
        text_id = shortuuid.uuid()[:8]

        logging.info(f"text: {text} ")
        #logging.info(f"text_embedding: {text_embedding} ")
        
        collection.add(
            documents=[text],
            metadatas=[{"type": "text", "source": pdf_name}],
            embeddings=[text_embedding.tolist()],
            ids=[text_id]
        )
        logging.info("################# images embeding store ##################")
        logging.info(f"images: {images} ")
        for img in images:
            logging.info(f"images desc: {img["description"]} ")
            logging.info(f"images path: {img["path"]} ")            
            img_id = shortuuid.uuid()[:8]
            img_path = img["path"]
            logging.info(f"img_path: {text} ")
            collection.add(
                documents=[text],
                metadatas=[{"type": "image", "source": pdf_name, "image_path": img_path}],
                embeddings=[text_embedding.tolist()],
                ids=[img_id]
            )

def process_pdf_upload(pdf_file):
    """
    จัดการการอัปโหลดและประมวลผล PDF
    """
    try:
        clear_vector_db()
        pdf_path = pdf_file.name
        pdf_name = os.path.basename(pdf_path)
        logging.info("#### start process pdf ####")
        
        content_chunks = extract_pdf_content(pdf_path)
        logging.info("#### start store in vector db ####")
        logging.info(f"content: {content_chunks}")

        store_in_chroma(content_chunks, pdf_name)
        
        return f"ประมวลผลและจัดเก็บ {pdf_name} สำเร็จ"
    except Exception as e:
        return f"เกิดข้อผิดพลาดในการประมวลผล PDF: {str(e)}"

def clear_vector_db():
    try:
        chroma_client.clear_system_cache()
       # Clear existing collection to avoid duplicates
        chroma_client.delete_collection(name="pdf_data_thai")
        global collection
        collection = chroma_client.create_collection(name="pdf_data_thai")

    except Exception as e:
        return f"เกิดข้อผิดพลาดในการล้างข้อมูล: {str(e)}"
    
def clear_vector_db_and_images():
    """
    ล้างข้อมูลใน Chroma vector database และไฟล์ในโฟลเดอร์ images
    """
    
    try:
        clear_vector_db()
        
        pdf_input.clear()
        if os.path.exists(TEMP_IMG):
            shutil.rmtree(TEMP_IMG)
            os.makedirs(TEMP_IMG, exist_ok=True)
        
        return "ล้างข้อมูลใน vector database และโฟลเดอร์ images สำเร็จ"
    except Exception as e:
        return f"เกิดข้อผิดพลาดในการล้างข้อมูล: {str(e)}"


def query_rag(question: str, n_results: int = 3):
    """
    ค้นหาในระบบ RAG และสร้างคำตอบแบบ streaming โดยใช้ Ollama
    """
    logging.info("####  RAG get Question ####")
    question_embedding = embed_text(question)
    is_summary=False

    results=[]
    # กรณี สรุป ใช้ ใช้ข้อมูล ทั้งหมด เพื่อ สรุป
    if "สรุป" in question or "summary" in question:        
        is_summary=True
        logging.info("####  Summary ####")
        results = collection.query(
            query_embeddings=[question_embedding.tolist()]       
        )
    else:
        logging.info("####  RAG ####")
        results = collection.query(
            query_embeddings=[question_embedding.tolist()],
            n_results=n_results
        )
    logging.info(f"##### results from vector: { results }")
    context_texts = []
    image_paths = []
    
    for doc, metadata in zip(results["documents"][0], results["metadatas"][0]):
        context_texts.append(doc)
        logging.info(doc)
        logging.info(f"metadata: {metadata}")
        # Regex pattern สำหรับค้นหา [ภาพ: ชื่อไฟล์.jpeg]
        pattern = r'\[ภาพ: [^\]]+\.jpeg\]'

        # ค้นหาทุกรูป แบบที่ตรงกับ ส่งเข้ามา
        imgs = re.findall(pattern, doc)
        print("---------------------------") 
        if imgs:
            image_paths.append(imgs)
            logging.info(f"img: {imgs}")

        print("---------------------------")
        if metadata:
            if metadata["type"] == "image":
                logging.info(f"image_path : { metadata["image_path"]}")
                image_paths.append(metadata['image_path'])

        #    image_paths.append(metadata["image_path"])
    

    context = "\n".join(context_texts)

    prompt = f"""จากบริบทต่อไปนี้ ตอบคำถาม: {question}

    บริบท:
    {context}

    ให้คำตอบที่ชัดเจนและกระชับเป็นภาษาไทย หากบริบทมีชื่อไฟล์รูปภาพ (เช่น [ภาพ: xxx.png]) ระบุว่ามีรูปภาพที่เกี่ยวข้อง"""

    ## กรณี สรุป ใช้ prompt นี้
    if (is_summary):
        prompt=f"""คุณคือผู้ช่วยในการสรุปข้อมูลจากเอกสาร โดยเน้นให้เข้าใจง่าย ใช้ภาษาที่ชัดเจน เหมาะกับผู้อ่านทั่วไป  
กรุณาสรุปข้อมูลด้านล่างให้เป็นภาษาธรรมดา ไม่ใช้ศัพท์เฉพาะมากเกินไป และจัดลำดับความสำคัญให้ผู้อ่านเข้าใจภาพรวม

[เนื้อหาจากเอกสารที่ดึงมาเป็นบริบท]  
    {context}

โปรดสรุปข้อมูลนี้ในรูปแบบข้อ ๆ หรือย่อหน้า ให้กระชับ ชัดเจน และครบถ้วน เป็นภาษาไทย"""


    logging.info("##############  Augmented prompt #################")
    logging.info(f"promt: {prompt}")
    logging.info("##############  Augmented prompt #################")
    stream = ollama.chat(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature":0.4 },
        stream=True
    )
    
    return stream, context_texts, image_paths

def user(user_message: str, history: List[Dict]) -> Tuple[str, List[Dict]]:
    """
    จัดการ input ของผู้ใช้และเพิ่มลงในประวัติการแชท
    """
    return "", history + [{"role": "user", "content": user_message}]

def chatbot_interface(history: List[Dict]):
    """
    อินเทอร์เฟซแชทบอทแบบ streaming
    """
    user_message = history[-1]["content"]
    stream, context_texts, image_paths = query_rag(user_message)
    


    history.append({"role": "assistant", "content": ""})
    full_answer=""
    for chunk in stream:
        content = chunk["message"]["content"]
        full_answer += content 
        history[-1]["content"] += content
        #logging.info(f"content: {content}")
        yield history
    
    #if context_texts:
    #    history[-1]["content"] += "\n\nบริบทที่เกี่ยวข้อง:\n" + "\n".join(context_texts)
    #    yield history
    
    if image_paths:
        history[-1]["content"] += "\n\nรูปภาพที่เกี่ยวข้อง:"

        best_image= find_best_image_for_answer(full_answer)
        yield history
        print("//////////////////////////")
        print(f"full_answer: {full_answer}") 
        print(f"img_paths : {image_paths}") 
        print(f"best_image : {best_image}") 
        print("//////////////////////////")
        for img_path in image_paths:
            
            
            img_file = image_cleanup(img_path)
            print("*********************************")
            logger.info(f"img_path: {img_file}")
            print("*********************************")
            for img in img_file:
                if os.path.exists(img):
                        image = Image.open(img)
                        buffered = io.BytesIO()
                        image.save(buffered, format="PNG")
                        image_response = f"![Image](data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode()})"

                        history.append({"role": "assistant", "content": image_response })
                        yield history

def image_cleanup(img_path):
    #['[ภาพ: 4mnMEfhS.jpeg]']  to ['/data/images/4mnMEfhS.jpeg' ] 
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print(img_path) 
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")   
   
    cleaned_img = []
   
    for item in img_path:
        if "ภาพ" in img_path :
            # แทนที่ '[ภาพ: ' ด้วยสตริงว่าง
            temp_item = item.replace('[ภาพ: ', '')
            # ลบ ']' ออกจากท้าย
            cleaned_item = temp_item.strip(']')
            cleaned_img.append(f"{TEMP_IMG}/{cleaned_item}")  # add file location 
        else:
            cleaned_item = img_path
            cleaned_img.append(f"{img_path}")  # add file location 

    unique_list = list(set(cleaned_img))  
    print(unique_list)
    return unique_list

def find_best_image_for_answer(answer: str) -> tuple:
    """
    ค้นหาภาพที่มีความเหมือนสูงสุดกับคำตอบ
    """
    answer_embedding = embed_text(answer)
    
    logging.info(f"answer: { answer }")
    # Query only images from Chroma
    results = collection.query(
        query_embeddings=[answer_embedding.tolist()],
        n_results=10,  # Get more results to ensure we have enough images
        where={"type": "image"}        
    )
    print("$$$$$$$$$$$$$$$$$")
    print(f"result: { results }")
    image_candidates = []
    
    #for metadata in zip(results["metadatas"][0]):
    #    similarity = 1 - cosine(answer_embedding, np.array())
    #    image_candidates.append({
    #        "path": metadata["image_path"],
    #        "description": metadata["image_description"],
    #        "markdown": metadata["image_markdown"],
    #        "similarity": similarity
    #    })
    
    # Select image with highest similarity
    selected_image = max(image_candidates, key=lambda x: x["similarity"]) if image_candidates else None
    
    if selected_image:
        return [selected_image["path"]], [selected_image["description"]], [selected_image["markdown"]]
    return [], [], []

# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# แชทบอท PDF: Multi-Modal RAG (ภาษาไทย)")
    
    with gr.Tab("แอดมิน - อัปโหลด PDF"):
        pdf_input = gr.File(label="อัปโหลดไฟล์ PDF")
        upload_button = gr.Button("ประมวลผล PDF")
        clear_button = gr.Button("ล้างข้อมูล")
        upload_output = gr.Textbox(label="สถานะการอัปโหลด")
        upload_button.click(
            fn=process_pdf_upload,
            inputs=pdf_input,
            outputs=upload_output
        )
        clear_button.click(
            fn=clear_vector_db_and_images,
            inputs=None,
            outputs=upload_output,
            queue=False
        )
    
    with gr.Tab("แชท"):
        chatbot = gr.Chatbot(type="messages")
        msg = gr.Textbox(label="ถามคำถามเกี่ยวกับ PDF")
        clear_chat = gr.Button("ล้าง")
        msg.submit(
            fn=user,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot],
            queue=False
        ).then(
            fn=chatbot_interface,
            inputs=chatbot,
            outputs=chatbot
        )
        clear_chat.click(lambda: [], None, chatbot, queue=False)

if __name__ == "__main__":
    demo.launch()