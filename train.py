import time
import tensorflow as tf
from tensorflow.keras import optimizers, callbacks


gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

def lr_schedule(epoch):
    initial_lr = 0.001
    if epoch > 75:
        return initial_lr * 0.01
    elif epoch > 50:
        return initial_lr * 0.1
    elif epoch > 25:
        return initial_lr * 0.5
    return initial_lr

def train_model(model, x_train, y_train, x_val, y_val, output_dir='.', use_augmentation=True):
    print("Training the model...")
    
    # Compile the model
    opt = optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Define callbacks
    checkpoint_cb = callbacks.ModelCheckpoint(f'{output_dir}/best_model.h5', 
                                             save_best_only=True, 
                                             monitor='val_accuracy')
    
    early_stopping_cb = callbacks.EarlyStopping(monitor='val_loss', 
                                               patience=10,
                                               restore_best_weights=True)
    
    lr_scheduler = callbacks.LearningRateScheduler(lr_schedule)
    
    # Train the model
    start_time = time.time()
    
    if use_augmentation:
        # Data augmentation
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            zoom_range=0.1
        )
        datagen.fit(x_train)
        
        history = model.fit(
            datagen.flow(x_train, y_train, batch_size=64),
            epochs=100,
            validation_data=(x_val, y_val),
            callbacks=[checkpoint_cb, early_stopping_cb, lr_scheduler]
        )
    else:
        history = model.fit(
            x_train, y_train,
            batch_size=64,
            epochs=100,
            validation_data=(x_val, y_val),
            callbacks=[checkpoint_cb, early_stopping_cb, lr_scheduler]
        )
    
    elapsed_time = time.time() - start_time
    print(f"Training completed in {elapsed_time/60:.2f} minutes")
    
    # Save the model
    model.save(f'{output_dir}/cifar10_model.h5')
    print(f"Model saved as '{output_dir}/cifar10_model.h5'")
    
    return model, history

def train_transfer_learning_model(model, x_train, y_train, x_val, y_val, output_dir='.'):
    print("Training the transfer learning model...")
    
    # Compile the model
    model.compile(optimizer=optimizers.Adam(learning_rate=0.0001),
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    
    # Define callbacks
    checkpoint_cb = callbacks.ModelCheckpoint(f'{output_dir}/best_transfer_model.h5', 
                                             save_best_only=True, 
                                             monitor='val_accuracy')
    
    early_stopping_cb = callbacks.EarlyStopping(monitor='val_loss', 
                                               patience=10,
                                               restore_best_weights=True)
    
    # Train the model
    start_time = time.time()
    
    history = model.fit(
        x_train, y_train,
        batch_size=64,
        epochs=50,
        validation_data=(x_val, y_val),
        callbacks=[checkpoint_cb, early_stopping_cb]
    )
    
    elapsed_time = time.time() - start_time
    print(f"Transfer learning training completed in {elapsed_time/60:.2f} minutes")
    
    # Save the model
    model.save(f'{output_dir}/cifar10_transfer_model.h5')
    print(f"Transfer learning model saved as '{output_dir}/cifar10_transfer_model.h5'")
    
    return model, history
