# Import des modules nécessaires
from http.server import BaseHTTPRequestHandler, HTTPServer  # Pour créer un serveur HTTP simple
import urllib.parse  # Pour parser les URLs et extraire les paramètres
import requests  # Pour télécharger le PDF depuis une URL
from PyPDF2 import PdfReader  # Pour lire le contenu des fichiers PDF
from io import BytesIO  # Pour manipuler le PDF en mémoire comme un fichier
import json  # Pour convertir les données en JSON
import re  # Pour utiliser des expressions régulières et extraire les données

# Définir le port sur lequel le serveur va écouter
PORT = 8000


# Création d'une classe qui gère les requêtes HTTP GET
class PDFtoJSONHandler(BaseHTTPRequestHandler):

    # Méthode appelée à chaque requête GET
    def do_GET(self):
        # Parser l'URL pour extraire le chemin et les paramètres
        parsed_path = urllib.parse.urlparse(self.path)
        # Extraire les paramètres GET dans un dictionnaire
        query_params = urllib.parse.parse_qs(parsed_path.query)

        # Vérifier si le paramètre 'url' est présent
        if 'url' not in query_params:
            # Si absent, envoyer une réponse 400 (Bad Request)
            self.send_response(400)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            # Envoyer le message d'erreur encodé en UTF-8
            self.wfile.write("Veuillez fournir un paramètre 'url' avec l'URL du PDF.".encode('utf-8'))
            return  # Sortir de la méthode

        # Récupérer la valeur du paramètre 'url'
        pdf_url = query_params['url'][0]

        try:
            # Télécharger le PDF depuis l'URL
            response = requests.get(pdf_url)
            # Vérifier si la requête a réussi, sinon générer une exception
            response.raise_for_status()

            # Lire le PDF en mémoire (BytesIO permet de traiter le contenu comme un fichier)
            pdf_file = BytesIO(response.content)
            # Créer un objet PdfReader pour lire le PDF
            reader = PdfReader(pdf_file)

            # Liste pour stocker toutes les lignes extraites du PDF
            lines = []
            # Parcourir toutes les pages du PDF
            for page in reader.pages:
                # Extraire le texte brut de la page
                text = page.extract_text()
                if text:
                    # Parcourir chaque ligne du texte
                    for line in text.splitlines():
                        # Nettoyer les espaces multiples et retirer les espaces en début/fin
                        clean_line = re.sub(r'\s+', ' ', line).strip()
                        # Ajouter la ligne propre à la liste si elle n'est pas vide
                        if clean_line:
                            lines.append(clean_line)

            # Dictionnaire pour stocker les données des devises
            data = {}
            # Parcourir toutes les lignes extraites
            for line in lines:
                # Utiliser une expression régulière pour détecter :
                # - code devise (3 lettres majuscules)
                # - nom de la devise (lettres, espaces, parenthèses, tirets)
                # - taux (nombre avec éventuellement un point décimal)
                match = re.match(r"^([A-Z]{3})\s+([A-Za-z\s\(\)\-]+)\s+([\d\.]+)$", line)
                if match:
                    # Extraire les groupes de l'expression régulière
                    code, name, rate = match.groups()
                    try:
                        # Ajouter la ligne valide au dictionnaire avec le code comme clé
                        data[code] = {"name": name.strip(), "rate": float(rate)}
                    except ValueError:
                        # Ignorer les lignes où le taux n'est pas convertible en float
                        continue

            # Vérifier si aucune donnée valide n'a été trouvée
            if not data:
                raise ValueError("Aucune ligne valide trouvée dans le PDF.")

            # Convertir le dictionnaire en JSON formaté (indenté pour lisibilité)
            json_output = json.dumps(data, indent=2, ensure_ascii=False)

            # Envoyer la réponse HTTP 200 (OK)
            self.send_response(200)
            # Définir le type de contenu de la réponse comme JSON UTF-8
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            # Envoyer le JSON encodé en UTF-8
            self.wfile.write(json_output.encode('utf-8'))

        except Exception as e:
            # Si une erreur se produit, envoyer une réponse 500 (Internal Server Error)
            self.send_response(500)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            # Envoyer le message d'erreur encodé en UTF-8
            self.wfile.write(f"Erreur lors du traitement du PDF : {e}".encode('utf-8'))


# Lancer le serveur si ce script est exécuté directement
if __name__ == "__main__":
    # Créer le serveur HTTP sur localhost et le port défini
    server = HTTPServer(('localhost', PORT), PDFtoJSONHandler)
    print(f"Serveur démarré sur http://localhost:{PORT}")
    # Démarrer la boucle infinie qui écoute les requêtes
    server.serve_forever()
