import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import pandas as pd
import numpy as np
from pymongo import MongoClient
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/inventory/optimization")
def get_inventory_optimization():
    # -------------------------------------------------
    # MongoDB Connection
    # -------------------------------------------------
    # Use environment variable for security; provide a default if needed
    connection_string = os.getenv("MONGO_CONNECTION_STRING")
    client = MongoClient(connection_string)
    db = client["InvenX"]

    # -------------------------------------------------
    # Retrieve Inventory Data
    # -------------------------------------------------
    inventory_collection = db["inventory"]
    inventory_docs = list(inventory_collection.find({}, {
        "_id": 0,
        "warehouse_id": 1,
        "product_id": 1,
        "current_stock": 1,
        "min_stock": 1,
        "max_stock": 1,
        "last_updated": 1,
        "forecast_value": 1
    }))
    inventory_df = pd.DataFrame(inventory_docs)
    if inventory_df.empty:
        raise HTTPException(status_code=404, detail="No inventory data found.")

    # -------------------------------------------------
    # Retrieve Warehouse Data (for ordering_cost)
    # -------------------------------------------------
    warehouse_collection = db["warehouse"]
    warehouse_docs = list(warehouse_collection.find({}, {
        "_id": 1,
        "ordering_cost": 1
    }))
    warehouses_df = pd.DataFrame(warehouse_docs)
    if not warehouses_df.empty and "_id" in warehouses_df.columns:
        warehouses_df = warehouses_df.rename(columns={"_id": "warehouse_id"})
    else:
        # Warn and use default ordering cost
        print("Warning: Warehouse Data is empty; using default ordering cost.")

    # -------------------------------------------------
    # Retrieve Warehouse Transfers Data (for transfer cost)
    # -------------------------------------------------
    warehouse_transfers_collection = db["warehouse_transfers"]
    warehouse_transfers_docs = list(warehouse_transfers_collection.find({}, {
        "_id": 0,
        "source_warehouse_id": 1,
        "destination_warehouse_id": 1,
        "transfer_cost": 1,
        "currency": 1,
        "estimated_transfer_time": 1
    }))
    warehouse_transfers_df = pd.DataFrame(warehouse_transfers_docs)

    # -------------------------------------------------
    # Merge Inventory with Warehouse Data to Include Ordering Cost
    # -------------------------------------------------
    default_ordering_cost = 2000.0  # Default ordering cost if warehouse data is missing
    if warehouses_df.empty or "ordering_cost" not in warehouses_df.columns:
        inventory_df["ordering_cost"] = default_ordering_cost
    else:
        inventory_df = pd.merge(inventory_df, warehouses_df[["warehouse_id", "ordering_cost"]],
                                  on="warehouse_id", how="left")
        inventory_df["ordering_cost"] = inventory_df["ordering_cost"].fillna(default_ordering_cost)

    # -------------------------------------------------
    # Configuration: Cost Parameters and Safety Factor
    # -------------------------------------------------
    safety_factor = 0.2  # 20% safety factor

    # -------------------------------------------------
    # Step 1: Compute Dynamic Safety Stock and Difference
    # -------------------------------------------------
    inventory_df['dynamic_min_stock'] = (inventory_df['forecast_value'] * (1 + safety_factor)).round()
    inventory_df['diff'] = inventory_df['current_stock'] - inventory_df['dynamic_min_stock']

    # -------------------------------------------------
    # Step 2: Generate Individual Warehouse Recommendations
    # -------------------------------------------------
    individual_recommendations = {}
    for idx, row in inventory_df.iterrows():
        prod = row["product_id"]
        ware = row["warehouse_id"]
        current_stock = row["current_stock"]
        forecast_value = row["forecast_value"]
        dynamic_min = row["dynamic_min_stock"]
        diff = row["diff"]
        min_stock = row["min_stock"]
        max_stock = row["max_stock"]
        order_cost = row["ordering_cost"]

        rec_message = ""
        external_order_cost_val = None

        # Use dynamic safety stock to determine deficit/surplus.
        if current_stock < dynamic_min:
            deficit = dynamic_min - current_stock
            external_order_cost_val = order_cost * deficit
            rec_message += f"Needs to order {deficit:.0f} units externally (Cost: {external_order_cost_val:.2f} rupees). "
        else:
            surplus = current_stock - dynamic_min
            rec_message += f"Has excess of {surplus:.0f} units; will attempt internal transfer. "

        if current_stock < min_stock:
            rec_message += "Stock is below the fixed minimum threshold! "
        if current_stock > max_stock:
            rec_message += "Stock exceeds the maximum limit; consider reducing inventory. "

        individual_recommendations[(prod, ware)] = {
            "current_stock": current_stock,
            "forecast_value": forecast_value,
            "dynamic_min_stock": dynamic_min,
            "diff": diff,
            "external_order_cost": external_order_cost_val,
            "recommendation": rec_message
        }

    # -------------------------------------------------
    # Step 3: Generate Cross-Warehouse Transfer Recommendations
    # -------------------------------------------------
    transfer_recommendations = {}
    for prod, group in inventory_df.groupby("product_id"):
        surplus_list = []  # (warehouse_id, surplus_amount)
        deficit_list = []  # (warehouse_id, deficit_amount)
        for idx, row in group.iterrows():
            wh = row["warehouse_id"]
            diff_val = row["current_stock"] - row["dynamic_min_stock"]
            if diff_val > 0:
                surplus_list.append((wh, diff_val))
            elif diff_val < 0:
                deficit_list.append((wh, -diff_val))
        for d_idx, (def_wh, def_qty) in enumerate(deficit_list):
            while def_qty > 0 and surplus_list:
                candidate_transfers = []
                for s_idx, (surp_wh, surp_qty) in enumerate(surplus_list):
                    if surp_qty > 0:
                        transfer_qty = min(surp_qty, def_qty)
                        transfer_record = warehouse_transfers_df[
                            (warehouse_transfers_df["source_warehouse_id"] == surp_wh) &
                            (warehouse_transfers_df["destination_warehouse_id"] == def_wh)
                        ]
                        if not transfer_record.empty:
                            cost_per_unit = float(transfer_record.iloc[0]["transfer_cost"])
                        else:
                            cost_per_unit = 15.0  # default per-unit transfer cost
                        total_transfer_cost = cost_per_unit * transfer_qty
                        candidate_transfers.append((surp_wh, transfer_qty, total_transfer_cost, s_idx))
                if candidate_transfers:
                    best_candidate = min(candidate_transfers, key=lambda x: x[2])
                    best_surp_wh, transfer_qty, total_transfer_cost, best_idx = best_candidate
                    key = (prod, best_surp_wh, def_wh)
                    transfer_recommendations[key] = transfer_recommendations.get(key, 0) + transfer_qty
                    surplus_list[best_idx] = (best_surp_wh, surplus_list[best_idx][1] - transfer_qty)
                    def_qty -= transfer_qty
                else:
                    break
            if def_qty > 0:
                key = (prod, def_wh, "EXTERNAL")
                transfer_recommendations[key] = f"Order {def_qty:.0f} units externally for Warehouse {def_wh}."

    # -------------------------------------------------
    # Return the Combined Recommendations as JSON
    # -------------------------------------------------
    response = {
        "individual_recommendations": {str(k): v for k, v in individual_recommendations.items()},
        "transfer_recommendations": {str(k): v for k, v in transfer_recommendations.items()}
    }
    return JSONResponse(content=response)

if __name__ == "__main__":
    # Use Render's provided PORT or default to 8000
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
