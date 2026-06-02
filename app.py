import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
import numpy as np

# Cấu hình hiển thị trang web chế độ hiển thị rộng (Wide mode)
st.set_page_config(page_title="Dashboard Kinh Tế Xã Hội", layout="wide")
st.title("📊 HỆ THỐNG TRỰC QUAN HÓA & PHÂN TÍCH CHỈ SỐ KINH TẾ - XÃ HỘI")

# 1. TẢI DỮ LIỆU ĐÃ LÀM SẠCH VÀO HỆ THỐNG GIAO DIỆN
df_ds = pd.read_excel('du_lieu_sach_khoa_hoc.xlsx', sheet_name='DanSo_Clean')
df_kt = pd.read_excel('du_lieu_sach_khoa_hoc.xlsx', sheet_name='KinhTe_Clean')

# 2. XÂY DỰNG BỘ LỌC ĐỘNG (INTERACTIVE SLICER) TRÊN SIDEBAR
st.sidebar.header("Bộ Lọc Hệ Thống")
danh_sach_nam = df_ds['Nam'].unique()
nam_chon = st.sidebar.selectbox("Chọn Năm Phân Tích", danh_sach_nam)

# Lọc các tập dữ liệu con theo năm được lựa chọn
df_ds_filtered = df_ds[df_ds['Nam'] == nam_chon]
df_kt_filtered = df_kt[df_kt['Nam'] == nam_chon]

# 3. TẦNG HIỂN THỊ CHỈ SỐ SỐ LƯỢNG TỔNG QUÁT (KPI CARDS)
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Tổng Số Dân Toàn Xã", f"{df_ds_filtered['Tong_So_Dan'].sum():,}")
with col2:
    st.metric("Tổng Số Hộ Nghèo", f"{df_ds_filtered['So_Ho_Ngheo'].sum():,}")
with col3:
    st.metric("Diện Tích Lúa (Ha)", f"{df_kt_filtered['Dien_Tich_ha'].sum():,.1f}")
with col4:
    st.metric("Tổng Sản Lượng Lúa (Tấn)", f"{df_kt_filtered['San_Luong_Tinh_Toan'].sum():,.1f}")

st.markdown("---")

# 4. TẦNG ĐỒ HỌA TRỰC QUAN BIỂU ĐỒ TƯƠNG TÁC (CHARTS LAYER)
left_col, right_col = st.columns(2)

with left_col:
    st.subheader("Số lượng Hộ Nghèo phân bổ chi tiết theo từng Ấp")
    # Vẽ biểu đồ cột tương tác bằng Plotly Express
    fig_ngheo = px.bar(df_ds_filtered, x='Ten_Ap', y='So_Ho_Ngheo', 
                       labels={'So_Ho_Ngheo': 'Số hộ nghèo', 'Ten_Ap': 'Tên ấp'},
                       color='Ten_Ap', text_auto=True,
                       color_discrete_sequence=px.colors.qualitative.Pastel)
    st.plotly_chart(fig_ngheo, use_container_width=True)

with right_col:
    st.subheader("Tỷ lệ cơ cấu các Giống lúa được gieo trồng")
    # Vẽ biểu đồ hình bánh (Pie donut chart) thể hiện tỷ lệ phần trăm các giống lúa
    fig_lua = px.pie(df_kt_filtered, values='Dien_Tich_ha', names='Giong_Lua', 
                     hole=0.4, color_discrete_sequence=px.colors.sequential.Plotly3)
    st.plotly_chart(fig_lua, use_container_width=True)

# 5. TÍCH HỢP MODULE KHOA HỌC DỰ BÁO XU HƯỚNG SẢN LƯỢNG (MACHINE LEARNING MODULE)
# Gom nhóm tổng sản lượng lúa của toàn xã theo từng năm lịch sử để làm tập dữ liệu học cho máy
thong_ke_chuoi_thoi_gian = df_kt.groupby('Nam')['San_Luong_Tinh_Toan'].sum().reset_index()

X_train = thong_ke_chuoi_thoi_gian[['Nam']]  # Biến độc lập (Feature)
y_train = thong_ke_chuoi_thoi_gian['San_Luong_Tinh_Toan']  # Biến phụ thuộc (Target)

# Huấn luyện mô hình hồi quy tuyến tính
mo_hinh_lr = LinearRegression()
mo_hinh_lr.fit(X_train, y_train)

# Tính toán dự báo cho năm tiếp theo (Năm 2026)
nam_tiep_theo = np.array([[2026]])
ket_qua_2026 = mo_hinh_lr.predict(nam_tiep_theo)[0]

# Hiển thị kết quả dự đoán của thuật toán Học máy ra thanh công cụ trái
st.sidebar.markdown("---")
st.sidebar.subheader("🔮 Dự báo Học Máy (ML)")
st.sidebar.write(f"Dự báo tổng sản lượng lúa toàn xã năm **2026** sẽ đạt khoảng:")
st.sidebar.info(f"**{ket_qua_2026:,.2f} Tấn**")