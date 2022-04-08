	.file "trans.s"	@ file name: stage1.s
	.align	2

	.data
	.align 	2
buff:	.skip 	100		@ array of char to store input via get_line()

	.align 2
dat:				@ global variable dat of type struct try
	.word 0			@ member inchar of type int
	.word 0			@ member outchar of type int
	.word 0			@ member charct of type int
	.word 0			@ member wordct of type int
	.word 0			@ member linect of type int
	.word 0
	_inchar = 0		@ _inchar is offset for member inchar
	_outchar = 4	@ _outchar is offset for member outchar
	_charct = 8		@ _charct is offset for member charct
	_wordct = 12	@ _wordct is offset for member wordct
	_linect = 16	@ _linect is offset for the member linect
	_subct = 20		@ _subct is offset for member subct, the variable that stores the number of substitutions

	.section	.rodata

	.align	2
summary:
	.asciz 	"Summary:\n"		@ format string for print_summary function header

	.align	2
chars:
	.asciz	"\t%d characters\n"	@ format string to print charct

	.align	2
lines:
	.asciz	"\t%d lines\n"		@ format string to print linect

	.align	2
words:
	.asciz	"\t%d words\n"		@ format string to print wordct

	.align	2
char_subs:
	.asciz	"\t%d substitutions\n"		@ format string to print wordct
	.text
	
/* print_summary:
	1 argument: Address of data structure datpoint
	state changes: The values of charct, wordct, linect 
	printed on standard input with labels
	return:	None */	
print_summary:
	push 	{fp, lr}		@set up stack frame
	add 	fp, sp, #4
	sub	sp, sp, #32

	mov	r4, r0			@ value of datpoint is moved to r4
	str	r4, [fp, #-8]
	@ [fp, #-8] holds the value of datP
	
	ldr	r0, summaryP		@ begin call to print format string at summaryP
	bl	printf

	ldr	r0, charsP			@ begin call to print format string at charsP
	ldr	r1, [r4, #_charct]
	bl	printf

	ldr	r0, wordsP			@ begin call to print format string at wordsP
	ldr	r1, [r4, #_wordct]
	bl	printf

	ldr	r0, linesP			@ begin call to print format string at linesP
	ldr	r1, [r4, #_linect]
	bl	printf

	ldr	r0, char_subsP
	ldr	r1, [r4, #_subct]
	bl	printf

	sub	sp, fp, #4		@ collapse stack frame
	pop	{fp, pc}		@ restore values of fp and pc


	/* pointer variables for members in .rodata and .data sections */

	.align	2
summaryP:				@ pointer to summary format string 
	.word	summary

	.align	2
charsP:					@ pointer to chars format string
	.word	chars

linesP:					@ pointer to linesP format string
	.word	lines

	.align	2
wordsP:					@ pointer to wordsP format string
	.word	words

	.align	2
char_subsP:					@ pointer to wordsP format string
	.word	char_subs

/* translate:
	two arguments: pointer datpoint to struct dat, and 
	address of null-terminated buffer of input buff
	state changes: number of characters in buff is added to charct
	number of words in buff is added to wordct
	each newline character in buff increments linect
	return:	value of buff string after translation */	
translate:

	push 	{fp, lr}		@ setup stack frame
	add 	fp, sp, #4
	sub	sp, sp, #16

	mov	r4, r0			@ value of datpoint is moved to r4
	str	r4, [fp, #-8]
	mov	r5, r1			@ value of buff is moved to r5
	str	r5, [fp, #-12]
	@ [fp, #-8] holds the value of datP
	@ [fp, #-12] holds the value of buff

	ldr	r5, [r4, #_inchar]	@ initialize r5 as inchar	
	ldr	r6, [r4, #_outchar]	@ initialize r6 as outchar
	ldr	r7, [r4, #_charct]	@ initialize r7 as charct
	ldr	r8, [fp, #-16]		@ initialize r8 as indx
	mov	r0, #0
	@str	r0, [r4, #_subct]
	str	r0, [fp, #-20]
	@ [fp, #-20] holds the local variable in_word

	
	mov	r8, #0				@ r8 holds value 0
	str r8, [fp, #-16]
	@ [fp, #-16] and r8 hold the local variable indx

	push 	{r4, r5, r6, r7, r8 }	@ push (save) the registers r4 through r8

loop:	

	ldr	r1, [fp, #-16]	@ begin call to get_byte
	ldr	r0, [fp, #-12]		 
	bl	get_byte
	@ r0 holds the value of the byte at location indx
	mov	r1, #0			
	cmp	r0, r1				@ compares buff[index] to null
	beq	post_put			@ end loop
	
	ldr	r0, [fp, #-12]		@ begin call to getbyte(buff, indx)
	ldr	r1, [fp, #-16] 
	bl	get_byte
	@ r0 holds the character at buff[index]

	cmp	r0, #10				@ compare byte value with newline
	beq	incr_linect			@ if equal, branch to increment linect 

 	cmp	r0, #32				@ compare byte value with space								
	bne	line_ct_guards		@ if not equal, branch past linect increment
 	ldr	r1, [fp, #-20]		
	cmp	r1, #1				@ compare in_word to 1
	beq	chg_in_word_to_0	@ if in_word == 1, change so in_word = 0

line_ct_guards:
 	cmp	r0, #122		@ if ASCII character value is greater than ASCII code for 'z'
	bgt	post_linect		@ skip in_word change branch

 	cmp	r0, #64			@ if ASCII character value is less than ASCII code for 'A'
	ble	post_linect	 	@ skip in_word change branch

	cmp	r0, #91			@ if ASCII character value equals 91 through 96 (unwanted symbols)
	beq	post_linect		@ skip in_word change branch

	cmp	r0, #92			@ ditto comments for value 91 cmp
	beq	post_linect

	cmp	r0, #93			@ ditto comments for value 91 cmp
	beq	post_linect

	cmp	r0, #94			@ ditto comments for value 91 cmp
	beq	post_linect

	cmp	r0, #95			@ ditto comments for value 91 cmp
	beq	post_linect 

	cmp	r0, #96			@ ditto comments for value 91 cmp
 	beq	post_linect 

	ldr	r1, [fp, #-20]	@ begin call to chg_in_word_to_1
	cmp	r1, #0
	beq	chg_in_word_to_1	

post_linect:
	mov	r1, r5			@ if buff[index] != inchar	
	@ldr	r1, [r4, #_inchar]	@ alternate way to load r1
	cmp	r0, r1			
	bne	loopend				@ then increment indx
	
	ldr	r0, [fp, #-12]		@ begin call to put_byte(buff, indx, outchar)
	ldr	r1, [fp, #-16]
	mov	r2, r6			
	bl	put_byte			@ replaces indx byte by outchar val 

	mov	r2, #1				@ begin increment of value of substitutions
	ldr	r3, [r4, #_subct]	
	add	r3, r2, r3			
	str r3, [r4, #_subct]	@ subct is incremented and stored in [r4, #_subct]														

loopend:
	ldr	r8, [fp, #-16]	@ steps to increment indx
	mov	r2, #1
	add	r8, r8, r2		
	str	r8, [fp, #-16]	@ store incremented value in indx
	b	loop			@ return to loop
post_put:
	ldr	r2, [fp, #-16]	@ begin incrementing charct value by indx-1
	sub	r2, r2, #1		@ null byte doesn't count to charct, subtract						
	ldr	r1, [r4, #_charct]
	@ mov r1, r7		@ alternate way to load r1
	add	r1, r1, r2		
	str	r1, [r4, #_charct]		@ store incremented value in charct value

	ldr	r0, [fp, #-12]			@ return buff from translate scope

	pop	{r4, r5, r6, r7, r8}	@ restore values of r4, r5, r6, r7, r8
	sub	sp, fp, #4				@ collapse stack frame
	pop	{fp, pc}				@ restore values of fp and pc		

incr_linect:	@ increment the value stored in [r4, #_linect] by 1
	ldr	r1, [r4, #_charct]	@ increment charct when newline is encountered
	@ mov r1, r7			@ alternate way to load r1
	add	r1, r1, #1
	str	r1, [r4, #_charct]	@ store incremented value in charct

	ldr	r1, [r4, #_linect]	@ increment linecount when newline is encountered
	add	r1, r1, #1
	str	r1, [r4, #_linect]	@ store incremented value in linect

	b	post_linect			@ branch to post_linect

chg_in_word_to_0:			@ change value stored in in_word to 0
	mov	r1, #0	
	str	r1, [fp, #-20]		@ value of 0 is stored in in_word
	b	post_linect			@ branch to post_linect

chg_in_word_to_1:	@ change value stored in in_word to 1
	ldr	r1, [r4, #_wordct]	@ load cumulative wordct
	add	r1, r1, #1
	str	r1, [r4, #_wordct]	@ store incremented wordct

	mov	r1, #1		
	str	r1, [fp, #-20]		@ assign 1 to in_word

	b	post_linect			@ branch to post_linect

	
	.section	.rodata

	.align	2		@ used this for help writing program
debug:
	.asciz 	"debugging worked: incorrect format, enter in format:%%c %%c \n"	
	.text
	
/* gettrans:
	2 Arguments: address of data structure datpoint, and string buff representing input line
    State change: First character is assigned to inchar, second character assigned to outchar
	return - Integer 1 on success, integer 0 for incorrect input form*/
gettrans:

	push {fp, lr}		@ setting up stack
	add fp, sp, #4
	sub	sp, sp, #8

	mov	r4, r0			@ argument holding datpoint moved to r4
	str	r4, [fp, #-8]		
	mov	r5, r1			@ argument holding pointer to buff moved to r5
	str	r5, [fp, #-12]
	@ [fp, #-8] and r4 hold pointer value of datP
	@ [fp, #-12] and r5 holds pointer value of buffP

	ldr	r0, [fp, #-12]	@ begin call to get_byte
	mov	r1, #0
	bl	get_byte
	@ r0 holds byte at buff[0]		

	cmp	r0, #48			@ if first byte is 0
	beq	return0			@ then begin exit from gettrans with branch to return0

	cmp	r0, #32			@ if first byte is space
	beq	return0			@ then begin exit from gettrans with branch to return0
	
	ldr	r0, [fp, #-12]		@ begin call to get_byte
	mov	r1, #1
	bl	get_byte			
	@ r0 holds byte at buff[1]

	cmp	r0, #32				@ if second character is not a space
	bne 	return0			@ then begin exit from gettrans with branch to return0
	
	ldr	r0, [fp, #-12]		@ begin call to get_byte
	mov	r1, #2
	bl	get_byte		
	@ r0 holds byte stored at buff[2]

	cmp	r0, #48				@ if third character is 0
	beq	return0				@ then begin exit from gettrans with branch to return0
	
	ldr	r0, [fp, #-12]		@ begin call to get_byte
	mov	r1, #3
	bl	get_byte
	@ r0 holds byte stored at buff[3]

	cmp	r0, #10
	bne	return0				@ then begin exit from gettrans with branch to return0
	
	ldr	r0, [fp, #-12]		@ begin call to get_byte
	mov	r1, #4
	bl	get_byte
	@r0 holds  byte stored at buff[4]

	cmp	r0, #0				@ if 5th character is not null
	bne	return0				@ then begin exit from gettrans with branch to return0
	
	ldr	r0, [fp, #-12]		@ begin call to get_byte
	@mov	r0, r5			@ value of buff moved into r0
	
	mov 	r1, #0			@ copy 1st character into inchar
	bl 	get_byte			
	@ r0 holds the byte at buff[0]

	str 	r0, [r4, #_inchar]	@ inchar holds 1st char of buff

	ldr	r0, [fp, #-12]		@ begin call to get_byte
	@mov	r0, r5			@ load r0 with buff
	mov 	r1, #2			
	bl 	get_byte
	@ r0 holds the byte at buff[2]

	str r0, [r4, #_outchar]	@ outchar holds the byte at buff[2]

	mov	r0, #1			@ gettrans returns value 1
	b	end				@ branch to end

return0:	@ return 0 for input format error

	ldr	r0, debugP		@ begin call to print input error
	bl	printf			
	mov	r0, #0			@ return a 0 if any of the required
						@ conditions / formatting is not met

	b	end				@ branch to end

end:				
	sub	sp, fp, #4		@ collapse stack frame
	pop	{fp, pc}		@ restore values of fp, pc
	bx	lr				@ return to main
	
	.align	2
debugP:	
	.word 	debug
	
	.section .rodata
	.align	2
inputprompt:				@ format string inputprompt printf
	.asciz "Please enter a line of input up to 99 characters:\n"
	
	/*insert additional strings here!!!!*/

	.align 	2
printprompt:				@ format string for buff printf
	.asciz "%s"

	.align	2
printgettrans:				@ format string for gettrans printf
	.asciz 	"%c%c\n"
	@.asciz "gettrans returns the characters inchar %c and outchar %c\n" @used for testing / clarity

	.align	2
returnval:
	.asciz	"return is %d\n"


	/* string variable */
	
	.text
	.global	main			@ main is globally visible name
main:	
	push	{fp, lr}		@setting up the stack
	add	fp, sp, #4
	sub	sp, sp, #16		

	push	{r4}			@ save and initialize r4
	ldr	r4, datP			@ assign address of dat to r4
	@ r4 holds the address of struct dat

	str	r4, [fp, #-8]		@ stores pointer to dat in pointer datP
	@ [fp, #-8] is a pointer variable datP that points to dat
	
	mov	r1, #0				
	str	r1, [r4, #_charct]	@ character count is 0 at start
	str	r1, [r4, #_linect]	@ line count is 0 at start
	str	r1, [r4, #_wordct]	@ word count is 0 at start

	@ldr r0, inputpromptP	@ begin call to print input prompt
	@bl	printf
	@ return r0 is number of printed characters from printf

	ldr	r0, buffP		@ begin call to getline(buff, 100)
	mov	r1, #100	
	bl	get_line
	@ return r0 now holds the number of bytes that were read
	@ Note: -1 is returned if error is encountered
	
	ldr	r0, datP		@ begin call to gettrans(datpoint, buff)
	ldr	r1, buffP
	bl	gettrans
	@ r0 now holds the return value 1

	ldr	r1, [r4, #_inchar]	@ begin call to print buff after gettrans
	ldr	r2, [r4, #_outchar]
	ldr	r0, printgettransP
	bl	printf
	@ return r0 is number of printed characters from printf

while_getline_input:

	@ldr	r0, inputpromptP	@ begin call to print input prompt
	@bl	printf
	@ return r0 is number of printed characters from printf
	
	
	ldr	r0, buffP		@ begin call to get_line(buff, 100)
	mov	r1, #100	
	bl	get_line
	@ return r0 now holds the number of bytes that were read
	@ Note: -1 is returned if error is encountered

	cmp	r0, #0			@ if no bytes were read
	beq	end_while_loop	@ branch end_while_loop

	ldr	r0, datP		@ begin call to translate(datpoint, buff)
	ldr	r1, buffP
	bl	translate
	@ returns translated string buff

	mov	r1, r0				@ begin call to printprompt (prints buff after translate)
	ldr	r0, printpromptP	
	@ldr	r1, buffP		@alternate way (instead of mov r1, r0) to put buff in r1
	bl	printf
	
	b	while_getline_input		@ branch to while_getline_input, the beginning of the loop

end_while_loop:		@ occurs when no bytes read from input line

	ldr	r0, datP		@ begin call to print_summary(datpoint)
	bl	print_summary

	mov	r0, #0			@ return 0 from main	
	pop	{r4}			@ restore the value of r4 register
	sub	sp, fp, #4		@ collapse stack frame of main
	pop	{fp, pc}		@ restore the value of fp, pc registers



/* pointer variables for members in .rodata and .data sections */	

	.align	2
inputpromptP:
	.word	inputprompt

	.align	2
returnq:
	.word	returnval	

	.align 	2
printpromptP:
	.word	printprompt

	.align	2
printgettransP:
	.word	printgettrans

	.align	2
buffP:
	.word	buff
	.align	2
datP:
	.word	dat
