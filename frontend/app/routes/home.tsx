import type { Route } from "./+types/home";
import Navbar from "~/components/Navbar";
import About from "~/components/About";
import Implementation from "~/components/Implementation";
import Team from "~/components/Team";
import Support from "~/components/Support";
import Contact from "~/components/Contact";

export function meta({}: Route.MetaArgs) {
  return [
    { title: "Lenni" },
    { name: "description", content: "Welcome to React Router!" },
  ];
}

export default function Home() {
  return <main className="bg-[url('/images/bg-main.svg')] bg-cover">

    <Navbar />

    <section className="main-section">
      <div className="page-heading min-h-screen items-center px-6 py-40">
        <h1>Talk Freely. Heal Gently. <br /> Lenni is Here to Listen.</h1>
        <h2>Lenni is an AI-powered mental health counsellor designed to support you through anxiety, stress, and everyday challenges — confidentially and judgment-free.</h2>
      </div>
    </section>

    <About />
    <Implementation />
    <Team />
    <Support />

    <Contact />

  </main>
}
