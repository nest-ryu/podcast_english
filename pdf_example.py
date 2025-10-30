from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase import pdfmetrics

# Register Korean font (prevents text breakage)
pdfmetrics.registerFont(UnicodeCIDFont("HYSMyeongJo-Medium"))

# Output path
pdf_path_business = "/mnt/data/비즈니스 스몰토크 – 업무에 대해 이야기하기.pdf"

# Set up document
doc = SimpleDocTemplate(pdf_path_business, pagesize=A4)
styles = getSampleStyleSheet()

# Styles
styles.add(ParagraphStyle(name="KTitle", fontName="HYSMyeongJo-Medium", fontSize=22, alignment=TA_CENTER, leading=26, spaceAfter=20))
styles.add(ParagraphStyle(name="KSub", fontName="HYSMyeongJo-Medium", fontSize=16, alignment=TA_CENTER, leading=26, spaceAfter=12))
styles.add(ParagraphStyle(name="KBody", fontName="HYSMyeongJo-Medium", fontSize=13, leading=26, alignment=TA_LEFT, spaceAfter=14))

story = []

# Cover Page
story.append(Spacer(1, 150))
story.append(Paragraph("💼 Business Small Talk", styles["KTitle"]))
story.append(Paragraph("업무에 대해 이야기하기", styles["KSub"]))
story.append(Spacer(1, 20))
story.append(Paragraph("“How have you been?”", styles["KBody"]))
story.append(Paragraph("“The figures look good.”", styles["KBody"]))
story.append(Paragraph("“Let’s share a taxi.”", styles["KBody"]))
story.append(Spacer(1, 30))
story.append(Paragraph("🌿 Fluent Practice Series", styles["KBody"]))
story.append(PageBreak())

# Dialogue Section
story.append(Paragraph("🗣️ 대화 스크립트 | Talking about Work", styles["KSub"]))
dialogue = """A: Hi, I was hoping to see you. How have you been? How's the family?
B: Oh, hello, Mr. Campbell. I'm fine, and Jack’s doing well. How are you?
A: I'm fine, thanks. I got your report this morning. Thanks for that.
B: Are you joining the conference today?
A: Yes, I’m leaving at 4 p.m.
B: Good. Well, we can discuss this more then. But I think the figures are looking very good for this quarter.
A: Yes, me too. I’m planning to discuss the advertising budget at the conference. I don’t think we should continue with the TV advertising.
B: No, me neither. It’s far too expensive.
A: Well, let’s discuss this more at the conference. Maybe we can share a taxi there?
B: Yes, sure."""
story.append(Paragraph(dialogue.replace("\n", "<br/>"), styles["KBody"]))

# Translation
story.append(Spacer(1, 15))
story.append(Paragraph("📘 대화 해석 | Translation", styles["KSub"]))
translation = """A: 만나길 바랐어요. 어떻게 지내셨어요? 가족분들은요?
B: 오, 캠벨 씨. 잘 지내고 있어요. 잭도 잘 지내요. 당신은요?
A: 저도 잘 지냅니다. 오늘 아침에 보고서 잘 받았습니다. 고맙습니다.
B: 오늘 회의 참석하시죠?
A: 네, 오후 4시에 출발할 거예요.
B: 좋아요. 그때 더 얘기해보죠. 이번 분기 수치는 아주 좋아 보이네요.
A: 네, 저도 그렇게 생각해요. 이번 회의에서 광고 예산에 대해 이야기할 계획이에요. TV 광고는 계속하지 않는 게 좋을 것 같아요.
B: 저도 그래요. 너무 비싸죠.
A: 그럼 회의 때 더 이야기합시다. 택시 같이 타실래요?
B: 네, 좋아요."""
story.append(Paragraph(translation.replace("\n", "<br/>"), styles["KBody"]))

# Expressions
story.append(PageBreak())
story.append(Paragraph("🔹 주요 표현 정리 | Key Expressions", styles["KSub"]))
expressions = """• I was hoping to see you. — 만나길 바랐어요.<br/>
• How have you been? — 어떻게 지내셨어요?<br/>
• How’s the family? — 가족분들은 잘 지내세요?<br/>
• I got your report this morning. — 오늘 아침에 보고서 받았어요.<br/>
• The figures are looking good. — 수치가 좋아 보이네요.<br/>
• Advertising budget — 광고 예산<br/>
• Me neither. — 나도 그래요 (부정문 동의)<br/>
• Far too expensive — 너무 비싸다<br/>
• Share a taxi / Split a cab — 택시를 같이 타다 / 비용을 나누다"""
story.append(Paragraph(expressions, styles["KBody"]))

# Summary
story.append(Spacer(1, 20))
story.append(Paragraph("🧩 핵심 포인트 요약 | Key Takeaways", styles["KSub"]))
summary = """- 비즈니스 스몰토크의 기본 구조: 인사 → 안부 → 일 관련 간단 대화 → 다음 약속 제안<br/>
- 자연스러운 연결어: Well, By the way, Anyway, Let's discuss this more later.<br/>
- 공식 자리에서도 너무 딱딱하지 않게, 부드럽고 자연스러운 어조 유지."""
story.append(Paragraph(summary, styles["KBody"]))

# Mission
story.append(Spacer(1, 20))
story.append(Paragraph("🎯 하루 미션 | Daily Mission", styles["KSub"]))
mission = """1️⃣ 오늘 'How have you been?'에 자연스럽게 대답해보기.<br/>
2️⃣ 동료에게 'Let's share a taxi.' 문장 직접 써보기.<br/>
3️⃣ 'The figures look good this quarter.' 문장으로 상황 문장 만들기."""
story.append(Paragraph(mission, styles["KBody"]))

# Build document
doc.build(story)

pdf_path_business
