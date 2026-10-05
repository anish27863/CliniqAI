import type { Metadata } from "next";
import { Inter, Space_Grotesk, IBM_Plex_Mono } from "next/font/google";
import "./globals.css";

const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
});

const spaceGrotesk = Space_Grotesk({
  variable: "--font-space-grotesk",
  subsets: ["latin"],
});

const ibmPlexMono = IBM_Plex_Mono({
  weight: ["500"],
  variable: "--font-ibm-plex-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "CliniqAI - Diagnostic Intelligence",
  description: "AI-powered medical scan diagnosis and attention mapping.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${inter.variable} ${spaceGrotesk.variable} ${ibmPlexMono.variable} font-sans antialiased bg-[#FAFBFC] text-[#0F1B2D] min-h-screen flex flex-col`}
      >
        {children}
      </body>
    </html>
  );
}
