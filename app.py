import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow import keras
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import subprocess
from collections import OrderedDict
import paramiko
import shutil

with st.sidebar: 
    image = Image.open(r"C:\Users\TRUNG QUAN\Downloads\Streamlit\hand.png")
    st.image(image)
    st.title("Thiết bị chuyển ngữ hỗ trợ người câm điếc trong giao tiếp")
    choice = st.radio("Tùy chọn", ["Giới thiệu","Lấy dữ liệu","Huấn luyện mô hình"])

# process = None
# def terminate_process():
#     if process:
#         process.send_signal(signal.SIGINT)


if choice == "Giới thiệu":
    st.info("Dự án tham gia cuộc thi Khoa học kỹ thuật cấp Tỉnh 2023-2024.")
    st.info("Tác giả: Cao Trung Quân.")



if choice == "Lấy dữ liệu":
    st.title("Lấy dữ liệu")
    st.info("Chú ý: Hãy bấm nút bật Server khi lấy dữ liệu!")
    sign = st.text_input("Đặt tên cho cử chỉ:")
    command =  "python udp-data-collect.py -d " + sign + " -l " + sign 
    if st.button("Lấy dữ liệu"):
        process = subprocess.Popen(command, shell=True)
    if st.button("Hoàn thành"):

        folder_path_data = 'C:\\Users\\TRUNG QUAN\\Downloads\\Streamlit\\' + sign

        for index in range(10):
            # Define the augmentation parameters
            augmentation_factor = 2  # Number of augmented samples to generate
            noise_std = 0.01  # Standard deviation of the noise

            # Define the column indices to exclude from augmentation
            columns_to_exclude = [2,3,4,8,9,10,14,15,16,20,21,22,26,27,28,32,33,34,38,39,40,45,46,47,51,52,53,57,58,59,63,64,65,69,70,71] # Columns to exclude (zero-based index)
            #[1,2,3,7,8,9,13,14,15,19,20,21,25,26,27,31,32,33,37,38,39,44,45,46,50,51,52,56,57,58,62,63,64,68,69,70]
            #[2,3,4,8,9,10,14,15,16,20,21,22,26,27,28,32,33,34,38,39,40,45,46,47,51,52,53,57,58,59,63,64,65,69,70,71]
            # Iterate over the CSV files in the subfolder
            for root, dirs, files in os.walk(folder_path_data):
                for file in files:
                    if file.endswith('.csv'):
                        # Read the original CSV file
                        file_path = os.path.join(root, file)
                        original_data = pd.read_csv(file_path)

                        # Apply noise augmentation
                        augmented_data = original_data.copy()

             
                        #for _ in range(augmentation_factor):
                        noise = np.random.normal(-noise_std, noise_std)
                        #print(len(noise))
                        try:
                            for i in columns_to_exclude:
                                augmented_data.iloc[:, i] = augmented_data.iloc[:, i].astype(float) + noise


                            # Create a new file name for the augmented data
                            new_file_name = os.path.splitext(file)[0] + '1.csv'
                            new_file_path = os.path.join(root, new_file_name)

                            # Save the augmented data to a new CSV file
                            augmented_data.to_csv(new_file_path, index=False)

                            print(f"Augmented data saved to: {new_file_path}")
                        except:
                            pass

        source_folder = 'C:\\Users\\TRUNG QUAN\\Downloads\\Streamlit\\' + sign
        destination_folder =  'C:\\Users\\TRUNG QUAN\\Downloads\\Streamlit\\k'

        if os.path.isdir(source_folder) and os.path.isdir(destination_folder):
            try:
                shutil.move(source_folder, destination_folder)
                # st.success("Folder moved successfully.")
            except Exception as e:
                st.error(f"An error occurred while moving the folder: {str(e)}")
        else:
            st.error("Invalid folder paths.")

        st.success("Đã lấy dữ liệu thành công!")



if choice == "Huấn luyện mô hình":
    st.title("Huấn luyện mô hình")

    st.info("Chú ý: Hãy bấm nút tắt Server khi huấn luyện mô hình!")
    folder_path = st.text_input("Hãy nhập đường dẫn đến thư mục")
    if folder_path:
        # st.success("Chọn thư mục thành công!")
        st.write("Đang tải lên dữ liệu của bạn...")

        if 'Cập nhật' not in st.session_state:
            st.session_state['Cập nhật'] = False

        x = []
        y = []
        for root, dirs, files in os.walk(folder_path):
            for file_name in files:
                if file_name.endswith(".csv"):
                    file_path = os.path.join(root, file_name)
                    label = file_name.split(".")[0]  # Extract the label from the file name

                    df = pd.read_csv(file_path)

                    # Process the data and extract features
                    # Assuming your first column is the timestamp and the remaining columns are the features
                    timestamps = df.iloc[:, 0]
                    features = df.iloc[:, 1:]  # Adjust the column index as per your data structure
                    feature_array = features.values.flatten()

                    if feature_array.dtype == np.float64 and feature_array.shape == (7400,):

                        # print(features.values.)

                        # Add the features and label to the lists
                        x.append(features.values.flatten())
                        y.append(label)
                    else:
                        #print("errorrrrrrrrrrrrrrrrrrrrrrrrrrr")
                        print("Error: Invalid features")
                        print(file_path)
                        #os.remove(file_path)

        label_list = list(OrderedDict.fromkeys(y))
        num_classes = len(label_list)

        # Convert the lists into numpy arrays
        x = np.array(x)
        y = np.array(y)

        # Perform label encoding
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

        # Split the data into training and testing sets
        test_size = 0.2  # Percentage of data to use for testing (adjust as needed)
        x_train, x_test, y_train, y_test = train_test_split(x, y_encoded, test_size=test_size, random_state=42)

        x_train = keras.utils.normalize(x_train, axis=1)
        x_test = keras.utils.normalize(x_test, axis=1)

        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)

        # print(x_train[0].shape)
        input_shape = x_train[0].shape

        st.success("Đã tải dữ liệu thành công! Đang chuyển sang bước Huấn luyện mô hình.")
        ### TRAIN MODEL


        st.write("Đang huấn luyện mô hình máy học...")

        model = keras.Sequential([
            keras.layers.Dense(30, activation='relu', input_shape=input_shape),
            keras.layers.Dense(30, activation='relu'),
            
            keras.layers.Dense(num_classes, activation='softmax')  # Change activation to softmax
        ])

        # Compile the model
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


        print("Training model ...")

        # Train the model
        epochs = 30
        losses = []
        accuracies = []
        for epoch in range(epochs):
            # Perform model training for each epoch
            history = model.fit(x_train, y_train)
            
            # Display the epoch information in Streamlit
            losses.append(history.history['loss'][0])
            accuracies.append(history.history['accuracy'][0])

            st.write(f"Epoch {epoch+1}/{epochs} hoàn thành:")
            st.write(f"Sai số: {losses[-1]}, Độ chính xác: {accuracies[-1]}")
            st.write()

        # Evaluate model on test set
        score = model.evaluate(x_test, y_test, verbose=0)

        st.success("Mô hình đã được huấn luyện thành công!")
        st.write(f"Sai số: {score[0]}")
        st.write(f"Độ chính xác: {score[1]}")

        # Save the Keras model (If it's ok)
        model.save('my_model.h5')

        # Convert the Keras model to TensorFlow Lite format
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()

        # Save the TensorFlow Lite model
        with open('my_model.tflite', 'wb') as f:
            f.write(tflite_model)

        # Convert model to float16
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]

        tflite_fp16_model = converter.convert()
        # Save the TensorFlow Lite float16 model
        with open('my_model.tflite', 'wb') as f:
            f.write(tflite_model)

        # Save the label_list as a text file if it doesn't exist
        # if not os.path.exists('label_list.txt'):
        with open('label_list.txt', 'w') as f:
            for label in label_list:
                f.write(label + '\n')

        # Define the file paths on your computer and Raspberry Pi
        local_model_file = r'C:\Users\TRUNG QUAN\Downloads\KHKT 2024\CODE\code app\Streamlit\my_model.tflite'
        local_label_file = r'C:\Users\TRUNG QUAN\Downloads\KHKT 2024\CODE\code app\Streamlit\label_list.txt'
        remote_model_file = '/home/pi/Documents/KHKT_2024/my_model.tflite'
        remote_label_file = '/home/pi/Documents/KHKT_2024/label_list.txt'

        # Define the Raspberry Pi's IP address, username, and password
        raspberry_pi_ip = '10.42.0.1'
        username = 'pi'
        password = 'tnb'

        # Create an SSH client
        ssh_client = paramiko.SSHClient()
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        try:
            # Connect to the Raspberry Pi
            ssh_client.connect(raspberry_pi_ip, username=username, password=password)

            # Create an SFTP client over the SSH connection
            sftp_client = ssh_client.open_sftp()

            # Transfer the files
            sftp_client.put(local_model_file, remote_model_file)
            sftp_client.put(local_label_file, remote_label_file)

            print("Model on server are updated!")

            # Close the SFTP client
            sftp_client.close()

        finally:
            # Close the SSH client
            ssh_client.close()

        st.success("Đã cập nhật thành công vào thiết bị!")