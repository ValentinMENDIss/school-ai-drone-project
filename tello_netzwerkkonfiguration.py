from djitellopy import Tello

# WLAN-Daten
SSID = "FLY_GHO"
PASSWORT = "17715621"

tello = Tello()

print("Verbinde mit der Drohne...")
tello.connect(wait_for_state=False)

print("Setze WLAN-Konfiguration...")

antwort = tello.send_command_with_return(f"ap {SSID} {PASSWORT}")
print("Antwort der Drohne:", antwort)

print("Fertig.")
print("Drohne jetzt ausschalten und neu starten.")

tello.end()