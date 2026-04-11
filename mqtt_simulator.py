import paho.mqtt.client as mqtt
import json
import time
import random
import math

BROKER = "broker.hivemq.com"
PORT = 1883
TOPIC = "pilar/capteurs"

client = mqtt.Client()
client.connect(BROKER, PORT, 60)

print("✅ Simulateur MQTT démarré — envoi toutes les 2 secondes")
print(f"📡 Broker: {BROKER} | Topic: {TOPIC}")
print("Scénarios: [1] Normal  [2] Surchauffe  [3] Usure critique  [4] Panne")
print("Ctrl+C pour arrêter\n")

scenario = 1
t = 0

while True:
    t += 1

    if scenario == 1:  # Normal
        temp_air = round(random.uniform(298, 302), 1)
        temp_process = round(temp_air + random.uniform(9, 11), 1)
        vitesse = round(random.uniform(1400, 1600))
        couple = round(random.uniform(35, 45), 1)
        usure = round(min(100 + t * 0.1, 150), 1)

    elif scenario == 2:  # Surchauffe progressive
        temp_air = round(300 + t * 0.2, 1)
        temp_process = round(temp_air + 12 + t * 0.3, 1)
        vitesse = round(random.uniform(1400, 1600))
        couple = round(random.uniform(35, 45), 1)
        usure = round(100 + t * 0.5, 1)

    elif scenario == 3:  # Usure critique
        temp_air = round(random.uniform(298, 302), 1)
        temp_process = round(temp_air + random.uniform(9, 11), 1)
        vitesse = round(random.uniform(1300, 1700))
        couple = round(random.uniform(50, 70), 1)
        usure = round(min(180 + t * 2, 250), 1)

    elif scenario == 4:  # Panne imminente
        temp_air = round(303 + random.uniform(0, 2), 1)
        temp_process = round(316 + random.uniform(0, 3), 1)
        vitesse = round(random.uniform(2500, 3000))
        couple = round(random.uniform(65, 80), 1)
        usure = 245

    data = {
        "type": 1,
        "temp_air": temp_air,
        "temp_process": temp_process,
        "vitesse": vitesse,
        "couple": couple,
        "usure": usure,
        "timestamp": time.time(),
        "scenario": scenario
    }

    client.publish(TOPIC, json.dumps(data))
    scenarios_names = {1:"Normal", 2:"Surchauffe", 3:"Usure critique", 4:"Panne"}
    print(f"📤 [{scenarios_names[scenario]}] Temp:{temp_air}K | Speed:{vitesse}rpm | Wear:{usure}min | Torque:{couple}Nm")

    # Change scenario toutes les 30 secondes
    if t % 15 == 0:
        scenario = (scenario % 4) + 1
        print(f"\n🔄 Changement de scénario → {scenarios_names[scenario]}\n")

    time.sleep(2)