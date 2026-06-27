import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image
import os
from pathlib import Path
import torch
import m  
from torchvision import transforms as T
out_dict = {
    0: 'human',
    1: 'non_human'
}
print("Đang tải mô hình AI vào bộ nhớ, vui lòng đợi...")
try:
    main_model = m.ViT()
    checkpoint = torch.load('model.pth', map_location='cpu')
    main_model.load_state_dict(checkpoint["checkpoints"])
    main_model.eval()  
    print("Tải mô hình thành công!")
except Exception as e:
    print(f"LỖI: Không thể tải mô hình. Chi tiết: {e}")
def load_img(img_link) -> str:
    clean_path = Path(img_link)
    img = Image.open(clean_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    transform = T.Compose([
        T.Resize((224,224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225])
    ])
    img = transform(img)
    img = torch.unsqueeze(img, 0)
    with torch.no_grad():
        output = main_model(img)
    return out_dict[output.squeeze(0).argmax(dim=0).item()]
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")
class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Phần Mềm Nhận Diện Ảnh AI")
        self.geometry("600x500")
        self.image_path = None
        self.label_title = ctk.CTkLabel(self, text="Hệ Thống Phân Loại: Người / Không Phải Người", font=ctk.CTkFont(size=20, weight="bold"))
        self.label_title.pack(pady=(20, 10))
        self.image_frame = ctk.CTkLabel(self, text="Chưa có ảnh nào được chọn", width=400, height=300, fg_color="gray30", corner_radius=8)
        self.image_frame.pack(pady=10)
        self.button_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.button_frame.pack(pady=10)
        self.btn_browse = ctk.CTkButton(self.button_frame, text="Chọn Ảnh", command=self.browse_image)
        self.btn_browse.pack(side="left", padx=10)
        self.btn_process = ctk.CTkButton(self.button_frame, text="Xử Lý Ảnh", command=self.process_image, state="disabled")
        self.btn_process.pack(side="left", padx=10)
        self.label_result = ctk.CTkLabel(self, text="", font=ctk.CTkFont(size=18, weight="bold"))
        self.label_result.pack(pady=10)
    def browse_image(self):
        file_path = filedialog.askopenfilename(
            title="Chọn một bức ảnh",
            filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp;*.webp")]
        )
        if file_path:
            self.image_path = file_path
            
            img = Image.open(self.image_path)
            ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(400, 300))
            
            self.image_frame.configure(image=ctk_img, text="")
            self.btn_process.configure(state="normal")
            self.label_result.configure(text="")
    def process_image(self):
        if not self.image_path:
            return
        self.label_result.configure(text="Đang phân tích mô hình...", text_color="orange")
        self.update() 
        try:
            ket_qua = load_img(self.image_path)
            if ket_qua == "human":
                self.label_result.configure(text="KẾT QUẢ: ĐÂY LÀ NGƯỜI (HUMAN)", text_color="#00FF00") 
            else:
                self.label_result.configure(text="KẾT QUẢ: KHÔNG PHẢI NGƯỜI (NON-HUMAN)", text_color="#FF3333") 
        except Exception as e:
            self.label_result.configure(text="Lỗi trong quá trình xử lý!", text_color="red")
            messagebox.showerror("Lỗi", f"Có lỗi xảy ra khi phân tích ảnh:\n{str(e)}")
if __name__ == "__main__":
    app = App()
    app.mainloop()