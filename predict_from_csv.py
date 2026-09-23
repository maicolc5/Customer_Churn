from predict_new_customer import predict_csv

output_path = predict_csv(
    "data/new_customers.csv",
    "data/customer_predictions2.csv"
)

print(f"Predicciones guardadas en: {output_path}")