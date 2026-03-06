package utils

import (
	"crypto/ed25519"
	"encoding/base64"
	"encoding/json"
	"errors"
	"strings"
	"time"
)

var (
	ErrInvalidToken = errors.New("invalid token")
	ErrExpiredToken = errors.New("token expired")
)

type JWTHeader struct {
	Alg string `json:"alg"`
	Typ string `json:"typ"`
}

type JWTPayload struct {
	Sub string `json:"sub"`
	Exp int64  `json:"exp"`
	Iat int64  `json:"iat"`
}

// GenerateJWT creates a new EdDSA (Ed25519) JWT token.
func GenerateJWT(privKey ed25519.PrivateKey, subject string, duration time.Duration) (string, error) {
	header := JWTHeader{
		Alg: "EdDSA",
		Typ: "JWT",
	}
	headerBytes, _ := json.Marshal(header)
	headerEncoded := base64.RawURLEncoding.EncodeToString(headerBytes)

	now := time.Now()
	payload := JWTPayload{
		Sub: subject,
		Iat: now.Unix(),
		Exp: now.Add(duration).Unix(),
	}
	payloadBytes, _ := json.Marshal(payload)
	payloadEncoded := base64.RawURLEncoding.EncodeToString(payloadBytes)

	unsignedToken := headerEncoded + "." + payloadEncoded

	signature := ed25519.Sign(privKey, []byte(unsignedToken))
	signatureEncoded := base64.RawURLEncoding.EncodeToString(signature)

	return unsignedToken + "." + signatureEncoded, nil
}

// ValidateJWT validates an EdDSA JWT token and returns the payload.
func ValidateJWT(pubKey ed25519.PublicKey, tokenString string) (*JWTPayload, error) {
	parts := strings.Split(tokenString, ".")
	if len(parts) != 3 {
		return nil, ErrInvalidToken
	}

	unsignedToken := parts[0] + "." + parts[1]
	signature, err := base64.RawURLEncoding.DecodeString(parts[2])
	if err != nil {
		return nil, ErrInvalidToken
	}

	if !ed25519.Verify(pubKey, []byte(unsignedToken), signature) {
		return nil, ErrInvalidToken
	}

	payloadBytes, err := base64.RawURLEncoding.DecodeString(parts[1])
	if err != nil {
		return nil, ErrInvalidToken
	}

	var payload JWTPayload
	if err := json.Unmarshal(payloadBytes, &payload); err != nil {
		return nil, ErrInvalidToken
	}

	if time.Now().Unix() > payload.Exp {
		return nil, ErrExpiredToken
	}

	return &payload, nil
}
