import type { Metadata } from "next";
import { VoiceConsole } from "../components/voice-console";

export const metadata: Metadata = {
  title: "Acme Orchestrator Console",
  description: "Live and mock UI for the Sales and Support multi-agent voice demo.",
};

export default function Page() {
  return <VoiceConsole />;
}
