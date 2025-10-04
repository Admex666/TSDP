import { google } from 'googleapis';
import { NextResponse } from 'next/server';

export async function POST(request) {
  try {
    const { email } = await request.json();

    if (!email || !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
      return NextResponse.json({ error: 'Érvénytelen email' }, { status: 400 });
    }

    // Private key dekódolása
    let privateKey;
    try {
      privateKey = Buffer.from(
        process.env.GOOGLE_PRIVATE_KEY_BASE64 || '',
        'base64'
      ).toString('utf-8');
      
      // Tisztítás: távolítsuk el a felesleges whitespace-eket
      privateKey = privateKey.trim();
    } catch (e) {
      console.error('Private key dekódolási hiba:', e);
      return NextResponse.json({ error: 'Konfiguráció hiba' }, { status: 500 });
    }

    if (!process.env.GOOGLE_CLIENT_EMAIL || !privateKey || !process.env.GOOGLE_SHEET_ID) {
      console.error('Hiányzó environment változók');
      return NextResponse.json({ error: 'Konfiguráció hiba' }, { status: 500 });
    }

    // JWT auth (modernebb, deprecation warnings nélkül)
    const auth = new google.auth.JWT({
      email: process.env.GOOGLE_CLIENT_EMAIL,
      key: privateKey,
      scopes: ['https://www.googleapis.com/auth/spreadsheets'],
    });

    const sheets = google.sheets({ version: 'v4', auth });

    await sheets.spreadsheets.values.append({
      spreadsheetId: process.env.GOOGLE_SHEET_ID,
      range: 'A:C',
      valueInputOption: 'USER_ENTERED',
      requestBody: {
        values: [[
          new Date().toLocaleString('hu-HU', { timeZone: 'Europe/Budapest' }),
          email,
          'Landing Page'
        ]],
      },
    });

    return NextResponse.json({ success: true });
  } catch (error) {
    console.error('Teljes hiba:', error);
    return NextResponse.json({ 
      error: 'Szerver hiba',
      message: error.message
    }, { status: 500 });
  }
}