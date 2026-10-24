import base64
import json
import os
from typing import Any
from urllib import parse, request

from google.genai import types


class SpotifyHelper:
    """Small helper for Spotify searches and token retrieval."""

    def __init__(self) -> None:
        self.client_id = os.getenv("SPOTIFY_CLIENT_ID", "")
        self.client_secret = os.getenv("SPOTIFY_CLIENT_SECRET", "")
        self.base_url = "https://api.spotify.com/v1"
        self.token_url = "https://accounts.spotify.com/api/token"

    def _get_access_token(self) -> str:
        """Fetch an OAuth access token using the client credentials flow."""
        if not self.client_id or not self.client_secret:
            raise ValueError(
                "Spotify is not configured. Set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET."
            )

        auth_string = f"{self.client_id}:{self.client_secret}"
        auth_header = base64.b64encode(auth_string.encode("utf-8")).decode("ascii")

        data = parse.urlencode({"grant_type": "client_credentials"}).encode("utf-8")
        req = request.Request(
            self.token_url,
            data=data,
            headers={
                "Authorization": f"Basic {auth_header}",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            method="POST",
        )

        with request.urlopen(req, timeout=15) as response:
            payload = json.loads(response.read().decode("utf-8"))

        access_token = payload.get("access_token")
        if not access_token:
            raise RuntimeError(f"Spotify token request failed: {payload}")

        return access_token

    def search_tracks(self, query: str, limit: int = 5) -> dict[str, Any]:
        """Search Spotify for tracks and return a compact JSON-friendly response."""
        cleaned_query = (query or "").strip()
        if not cleaned_query:
            return {"error": "Spotify search query cannot be empty."}

        if not self.client_id or not self.client_secret:
            return {
                "error": (
                    "Spotify is not configured. Add SPOTIFY_CLIENT_ID and "
                    "SPOTIFY_CLIENT_SECRET to your environment."
                )
            }

        try:
            token = self._get_access_token()
            clean_limit = max(1, min(int(limit or 5), 10))
            encoded_query = parse.quote(cleaned_query)
            url = f"{self.base_url}/search?q={encoded_query}&type=track&limit={clean_limit}"

            req = request.Request(
                url,
                headers={"Authorization": f"Bearer {token}"},
            )

            with request.urlopen(req, timeout=15) as response:
                payload = json.loads(response.read().decode("utf-8"))

            tracks = payload.get("tracks", {}).get("items", [])
            results = []

            for item in tracks:
                artists = ", ".join(artist.get("name", "") for artist in item.get("artists", []))
                results.append(
                    {
                        "name": item.get("name"),
                        "artist": artists,
                        "album": item.get("album", {}).get("name"),
                        "uri": item.get("uri"),
                        "preview_url": item.get("preview_url"),
                    }
                )

            return {"tracks": results}

        except Exception as exc:  # pragma: no cover - network/error path
            return {"error": f"Spotify search failed: {exc}"}


def build_spotify_tool() -> types.Tool:
    """Return a Gemini tool definition for Spotify track search."""
    return types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="search_spotify_tracks",
                description="Search Spotify for tracks by artist or song title and return a short list of results.",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "query": types.Schema(
                            type=types.Type.STRING,
                            description="The artist name or song title to search for.",
                        ),
                        "limit": types.Schema(
                            type=types.Type.INTEGER,
                            description="Maximum number of results to return (default 5).",
                        ),
                    },
                    required=["query"],
                ),
            )
        ]
    )
