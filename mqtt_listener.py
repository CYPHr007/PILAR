import paho.mqtt.client as mqtt
import json
import requests

BROKER = "broker.hivemq.com"
PORT = 1883
TOPIC = "pilar/capteurs"

# Change avec ton URL Railway ou localhost
PILAR_URL = "https://pilar-production-a05f.up.railway.app/predire"

def on_connect(client, userdata, flags, rc):
    print(f"✅ Connecté au broker MQTT")
    client.subscribe(TOPIC)
    print(f"📡 En écoute sur: {TOPIC}\n")

def on_message(client, userdata, msg):
    try:
        data = json.loads(msg.payload.decode())
        print(f"📥 Reçu: Temp:{data['temp_air']}K | Speed:{data['vitesse']}rpm | Wear:{data['usure']}min")

        # Envoie à Pilar pour analyse
        response = requests.post(PILAR_URL, json=data, timeout=10)
        result = response.json()

        risk = result['probabilite']
        status = "🔴 ANOMALIE" if result['prediction'] == 1 else "✅ Normal"
        zones = ', '.join([z['nom'] for z in result['zones']]) if result['zones'] else 'aucune'

        print(f"🧠 Analyse: {status} | Risque: {risk}% | Zones: {zones}")
        if result['mail_envoye']:
            print(f"📧 Alerte email envoyée !")
        print()

    except Exception as e:
        print(f"❌ Erreur: {e}")

client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
client.connect(BROKER, PORT, 60)

print("🚀 Pilar MQTT Listener démarré...")
client.loop_forever()