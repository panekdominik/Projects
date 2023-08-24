import csv
import tensorflow as tf
from tensorboard.backend.event_processing import event_accumulator

def save_tensor_to_csv(tensor_data, csv_filename, header):
    with open(csv_filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Step', header])
        for step, tensor_value in tensor_data:
            csvwriter.writerow([step, tensor_value])

def convert_tensor_events_to_csv(log_dir, tensor_tag, csv_output_filename, header):
    # Load the TensorBoard log data
    tensor_data = []

    for tensor_event in tf.compat.v1.train.summary_iterator(log_dir):
        for value in tensor_event.summary.value:
            if value.tag == tensor_tag:
                step = tensor_event.step
                tensor_value = tf.make_ndarray(value.tensor)
                if tensor_value.ndim == 0: 
                    tensor_data.append((step, tensor_value))

    save_tensor_to_csv(tensor_data, csv_output_filename, header=header)
    print(f'Tensor data exported to CSV file: {csv_output_filename}')

def return_tags(tensor_data):

    ea = event_accumulator.EventAccumulator(tensor_data)
    ea.Reload()
    events = []
    for tensor_event in ea.Tags()['tensors']:
        events.append(tensor_event)
        
    print("Losses recorded during training: ", events)